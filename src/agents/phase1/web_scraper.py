"""Web scraper using trafilatura + ThreadPoolExecutor for parallel page scraping.

Enhanced with multiple cleaning strategies:
- HTML pre-processing to remove unwanted sections
- Enhanced trafilatura extraction
- boilerpy3 fallback for noisy pages
- Pattern-based post-processing to remove metadata
"""

import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import List, Dict, Any
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from boilerpy3 import extractors

from config.settings import settings

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Minimum chars for a page to be considered useful content
# Increased to filter out low-quality/short content that wastes tokens
MIN_CONTENT_LENGTH = 300


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for noisy libraries like boilerpy3)."""
    stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = stderr


def is_deep_page(url: str) -> bool:
    """Filter out home pages and shallow URLs. Keep only deep/article pages."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    # Home pages have empty or very short paths
    if not path or path in ("/index.html", "/index.htm", "/home"):
        return False
    # Must have at least one meaningful path segment
    segments = [s for s in path.split("/") if s]
    return len(segments) >= 1


def preprocess_html(html: str) -> str:
    """Remove unwanted HTML sections before extraction.

    Removes navigation, footers, sidebars, author sections, and other
    non-content elements to improve extraction quality.

    Args:
        html: Raw HTML string

    Returns:
        Cleaned HTML string
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove entire elements that are typically non-content
    unwanted_tags = ['nav', 'footer', 'aside', 'header', 'script', 'style', 'form']
    for tag in unwanted_tags:
        for element in soup.find_all(tag):
            element.decompose()

    # Remove elements by class/id patterns (VERY AGGRESSIVE - common metadata sections)
    unwanted_patterns = [
        'author', 'byline', 'metadata', 'meta', 'share', 'social',
        'comment', 'sidebar', 'related', 'recommend', 'newsletter',
        'subscription', 'footer', 'header', 'nav', 'ad', 'advertisement',
        'promo', 'banner', 'popup', 'modal', 'cookie', 'consent',
        'citation', 'reference', 'bibliography', 'footnote',
        'breadcrumb', 'pagination', 'tags', 'category',
        'trending', 'popular', 'latest', 'widget',
        'subscribe', 'follow', 'signup', 'login',
    ]

    for pattern in unwanted_patterns:
        # Remove by class
        for element in soup.find_all(class_=re.compile(pattern, re.I)):
            element.decompose()
        # Remove by id
        for element in soup.find_all(id=re.compile(pattern, re.I)):
            element.decompose()

    # AGGRESSIVE: Remove small paragraphs (often navigation or ads)
    for p in soup.find_all('p'):
        if p.get_text(strip=True) and len(p.get_text(strip=True)) < 50:
            p.decompose()

    # AGGRESSIVE: Remove lists that are likely navigation
    for ul in soup.find_all(['ul', 'ol']):
        # If list has many short items, likely navigation
        items = ul.find_all('li')
        if items:
            avg_length = sum(len(li.get_text(strip=True)) for li in items) / len(items)
            if avg_length < 30 and len(items) > 3:  # Short items, many of them
                ul.decompose()

    return str(soup)


def clean_article_content(text: str) -> str:
    """Remove common metadata and non-content sections from extracted text.

    Uses pattern matching to identify and remove:
    - Author information and bylines
    - Email addresses
    - Reference sections
    - Social sharing prompts
    - Newsletter subscriptions
    - Short metadata lines

    Args:
        text: Extracted text content

    Returns:
        Cleaned text with metadata removed
    """
    if not text:
        return text

    lines = text.split('\n')
    cleaned_lines = []
    skip_section = False

    # Patterns for lines to remove entirely
    author_patterns = [
        r'^(by|author|written by|posted by|published by|editor)[\s:]+.+$',
        r'^\s*@[\w\.-]+\s*$',  # Email addresses on their own line
        r'[\w\.-]+@[\w\.-]+\.\w+',  # Email addresses anywhere
        r'^(contact|email)[\s:]+.+$',
        r'^(date|published|updated)[\s:]+.+$',
        r'^(tags|categories)[\s:]+.+$',
    ]

    # Patterns for academic references/citations (AGGRESSIVE)
    reference_patterns = [
        r'^\s*\[\d+\]',  # Numbered references like [1], [118], etc.
        r'^\s*\d+\.\s+[A-Z]',  # Numbered list starting with capital letter (1. Author)
        r'^\s*[A-Z]\.\s+[A-Z]\.',  # Initials like "D. Mougenot"
        r'^\s*[A-Z][a-z]+\s+[A-Z]\.\s+[A-Z]',  # First Last Initial format
        r'\bvol\.\s*\d+',  # Volume numbers (vol. 15)
        r'\bpp\.\s*\d+',  # Page numbers (pp. 85-97)
        r'\bno\.\s*\d+',  # Issue numbers (no. 3-4)
        r'\bdoi:',  # DOI identifiers
        r'\barxiv:',  # ArXiv identifiers
        r',\s*\d{4}\.',  # Year in citations like ", 2024."
        r'https?://[^\s]+',  # URLs
        r'\(\w+\s+\d{4}\)',  # Date formats like (December 2024), (May 2024)
        r'\.\s+\d{4}\.',  # Author. 2024. format
        r'Retrieved from|Available at|Accessed on',  # Citation retrieval text
        r'et al\.',  # Et al in citations
        r'\b(The\s+)?(Verge|Reuters|New York Times|CNN|BBC|Guardian|Washington Post|Forbes)',  # News outlets
    ]

    # Section headers and UI elements to skip (VERY AGGRESSIVE)
    skip_section_headers = {
        'references', 'bibliography', 'citations', 'works cited', 'sources', 'footnotes',
        'about the author', 'author bio', 'author information', 'contributor',
        'related articles', 'related posts', 'you may also like', 'more from',
        'recommended reading', 'further reading', 'read more', 'continue reading',
        'share this', 'share on', 'follow us', 'connect with us', 'join us',
        'comments', 'leave a comment', 'post a comment', 'view comments',
        'subscribe', 'newsletter', 'sign up', 'join our', 'get updates',
        'advertisement', 'sponsored', 'promoted content', 'partner content',
        'table of contents', 'jump to', 'navigation', 'menu',
        'privacy policy', 'cookie policy', 'terms of service', 'cookies',
        'all rights reserved', 'copyright', '©',
        'previous article', 'next article', 'page ', 'of ',
        'click here', 'learn more', 'find out', 'discover',
        'download', 'get started', 'try now', 'free trial',
        'trending', 'popular', 'most read', 'editor picks',
    }

    # Navigation and UI noise patterns
    ui_noise_patterns = [
        r'^\s*(home|back|next|previous|menu|search|login|sign in|sign up|register)\s*$',
        r'^\s*page\s+\d+',  # Page numbers
        r'^\s*\d+\s*of\s*\d+',  # "1 of 10"
        r'click here|tap here|swipe',  # UI instructions
        r'^\s*(yes|no|ok|cancel|close|skip)\s*$',  # Button text
        r'accept (cookies|all)',  # Cookie notices
        r'managing your privacy',  # Privacy text
        r'^\s*[\d\W]{3,}\s*$',  # Lines with mostly numbers/symbols
    ]

    # Promotional/SEO spam patterns
    spam_patterns = [
        r"don't miss", r"limited time", r"act now", r"hurry",
        r"exclusive offer", r"special deal", r"discount",
        r"subscribe now", r"sign up today", r"join today",
        r"download our app", r"get the app",
        r"follow us on", r"like us on",
        r"rated \d+", r"\d+ stars", r"customer reviews",
    ]

    for line in lines:
        stripped = line.strip()
        line_lower = stripped.lower()

        # Keep empty lines for paragraph separation
        if not stripped:
            if cleaned_lines and cleaned_lines[-1]:  # Avoid multiple consecutive empty lines
                cleaned_lines.append('')
            continue

        # AGGRESSIVE FILTERING: Check multiple garbage patterns

        # 1. Check if line contains URLs (likely citation or navigation)
        if 'http://' in stripped or 'https://' in stripped:
            skip_section = True
            continue

        # 2. Check if this line looks like a reference/citation
        is_reference = any(re.search(pattern, stripped) for pattern in reference_patterns)
        if is_reference:
            skip_section = True  # Enter skip mode when we detect references
            continue

        # 3. Check for UI noise patterns
        is_ui_noise = any(re.search(pattern, line_lower) for pattern in ui_noise_patterns)
        if is_ui_noise:
            continue

        # 4. Check for spam/promotional patterns
        is_spam = any(re.search(pattern, line_lower) for pattern in spam_patterns)
        if is_spam:
            continue

        # 5. Check if this line is a section header we should skip
        is_skip_header = any(section in line_lower for section in skip_section_headers)
        if is_skip_header:
            skip_section = True
            continue

        # If we're in a skip section, check if we've reached new content
        if skip_section:
            # Stay in skip mode if we see reference-like content
            if is_reference:
                continue
            # A long line likely indicates we're back to main content
            # But make sure it doesn't look like a reference
            if len(stripped) > 100 and not any(kw in line_lower for kw in ['subscribe', 'follow', 'share']):
                # Double-check it's not a reference with vol/pp/etc
                if not any(re.search(pattern, stripped) for pattern in reference_patterns):
                    skip_section = False
                else:
                    continue
            else:
                continue

        # Remove lines matching author/metadata patterns
        if any(re.search(pattern, line_lower) for pattern in author_patterns):
            continue

        # Remove very short lines with colons (often metadata like "Date: ..." "Tags: ...")
        if len(stripped) < 30 and ':' in stripped:
            # Check if it looks like metadata
            before_colon = stripped.split(':')[0].strip().lower()
            metadata_keywords = ['by', 'author', 'date', 'time', 'posted', 'published',
                                'updated', 'tags', 'category', 'filed', 'source']
            if any(kw in before_colon for kw in metadata_keywords):
                continue

        # Remove lines with common social sharing prompts
        if any(phrase in line_lower for phrase in ['share this', 'tweet this', 'pin it', 'share on']):
            continue

        # Remove lines that have multiple citation indicators (likely bibliography entries)
        citation_indicator_count = sum(1 for pattern in reference_patterns if re.search(pattern, stripped))
        if citation_indicator_count >= 2:
            continue

        # AGGRESSIVE: Remove short lines that don't look like complete sentences
        # Short lines without proper punctuation are often navigation/UI elements
        if len(stripped) < 60:
            # Must end with sentence-ending punctuation or be a proper sentence fragment
            if not stripped[-1] in '.!?:"\')"' if stripped else True:
                # Exception: Allow if it looks like a meaningful heading (title case, no numbers)
                if not (stripped[0].isupper() and stripped.replace(' ', '').isalpha()):
                    continue

        # AGGRESSIVE: Remove lines that are mostly navigation (few words, many capitals/symbols)
        if len(stripped) < 100:
            words = stripped.split()
            if len(words) > 0:
                # If more than 50% of words are capitalized, likely navigation/menu
                capital_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
                if capital_ratio > 0.6 and len(words) < 10:
                    continue

        # AGGRESSIVE: Skip lines with repeated words (often SEO/spam)
        words = stripped.lower().split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:  # More than 50% repeated words
                continue

        # Keep the line
        cleaned_lines.append(line)

    # AGGRESSIVE: Remove duplicate/near-duplicate consecutive lines
    # (common in scraped content due to headers, footers, etc.)
    deduped_lines = []
    prev_line_normalized = ""

    for line in cleaned_lines:
        # Normalize by removing extra spaces and lowercasing
        normalized = ' '.join(line.lower().split())

        # Skip if exact duplicate of previous line
        if normalized == prev_line_normalized and normalized:
            continue

        # Skip if very similar to previous line (80% similarity threshold)
        if normalized and prev_line_normalized:
            # Simple similarity: check if one is substring of other
            if normalized in prev_line_normalized or prev_line_normalized in normalized:
                if len(normalized) > 20 and len(prev_line_normalized) > 20:  # Only for substantial lines
                    continue

        deduped_lines.append(line)
        prev_line_normalized = normalized

    # Join and clean up multiple consecutive empty lines
    result = '\n'.join(deduped_lines)
    result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines

    # AGGRESSIVE: Remove very short final result (likely just garbage)
    if len(result) < 100:
        return ""

    return result.strip()


def scrape_single_page(url: str) -> Dict[str, Any]:
    """Scrape a single page with enhanced cleaning. Retries once on failure.

    Uses a multi-stage approach:
    1. HTML pre-processing to remove unwanted sections
    2. trafilatura extraction with enhanced settings
    3. Fallback to boilerpy3 if needed
    4. Pattern-based post-processing to remove metadata

    Args:
        url: URL to scrape

    Returns:
        Dict with url, text, and success fields
    """
    for attempt in range(2):
        try:
            response = httpx.get(
                url,
                timeout=settings.WEB_SCRAPER_TIMEOUT,
                follow_redirects=True,
                headers=HEADERS,
            )
            response.raise_for_status()

            # Step 1: Pre-process HTML to remove unwanted sections
            cleaned_html = preprocess_html(response.text)

            # Step 2: Extract with trafilatura (enhanced settings)
            text = trafilatura.extract(
                cleaned_html,
                include_comments=False,
                include_tables=False,
                include_links=False,      # Remove link text
                include_images=False,     # Remove image references
                no_fallback=False,
                output_format="txt",
                favor_precision=True,     # Prioritize precision over recall
            )

            # Step 3: Fallback to boilerpy3 if trafilatura fails or returns very short text
            # Note: boilerpy3 can be fragile with certain HTML structures, so we silently fail
            if not text or len(text) < MIN_CONTENT_LENGTH:
                try:
                    with suppress_stderr():  # Suppress boilerpy3 error messages
                        extractor = extractors.ArticleExtractor()
                        text = extractor.get_content(response.text)
                except (AttributeError, ValueError, KeyError, Exception):
                    # boilerpy3 may fail on malformed HTML - silently continue with trafilatura result
                    pass

            # Step 4: Post-process to remove remaining metadata
            if text:
                text = clean_article_content(text)

            # Final validation
            if text and len(text) >= MIN_CONTENT_LENGTH:
                return {"url": url, "text": text, "success": True}

        except Exception as e:
            if attempt == 0:
                continue  # retry once

    return {"url": url, "text": "", "success": False}


def scrape_all_pages(urls: List[str]) -> List[Dict[str, Any]]:
    """Scrape all URLs in parallel using ThreadPoolExecutor.

    Filters out home pages before scraping.

    Args:
        urls: List of URLs to scrape.

    Returns:
        List of dicts with url, text, and success fields.
    """
    # Filter to deep pages only
    deep_urls = [u for u in urls if is_deep_page(u)]
    skipped = len(urls) - len(deep_urls)
    if skipped:
        print(f"   Skipped {skipped} home/shallow page URLs")

    results = []
    with ThreadPoolExecutor(max_workers=settings.WEB_SCRAPER_MAX_WORKERS) as executor:
        future_to_url = {executor.submit(scrape_single_page, u): u for u in deep_urls}
        for future in as_completed(future_to_url):
            results.append(future.result())
    return results
