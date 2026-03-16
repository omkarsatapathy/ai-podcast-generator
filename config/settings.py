"""Application configuration settings."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    #LLM settings
    QUERY_PRODUCER_MODEL: str = "gpt-4o"
    QUERY_PRODUCER_TEMPERATURE: float = 0.9

    # API Settings
    API_TITLE: str = "AI Podcast Generator"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Automated Multi-Speaker Podcast Generation Pipeline"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # External API Keys
    OPENAI_API_KEY: str = ""
    GOOGLE_SEARCH_API_KEY: str = ""
    GOOGLE_SEARCH_ENGINE_ID: str = ""
    GEMINI_API_KEY: str = ""

    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    TEMP_DIR: Path = DATA_DIR / "temp"

    # Processing Settings
    MIN_PODCAST_DURATION_SEC: int = 900  # 15 minutes
    MAX_PODCAST_DURATION_SEC: int = 1800  # 30 minutes
    NUM_SPEAKERS: int = 2

    #Google Web search settings
    SEARCH_RESULTS_PER_QUERY: int = 10

    #query Producer settings
    QUERY_GENERATION_TEMPERATURE: float = 0.7
    QUERY_PRODUCE_PER_TOPIC: int = 15

    # Web Scraper Settings
    WEB_SCRAPER_MAX_WORKERS: int = 50
    WEB_SCRAPER_TIMEOUT: int = 15  # seconds per request
    LINK_FAILURE_THRESHOLD: float = 0.3  # 30% failure triggers query rewrite
    MAX_QUERY_REWRITE_ATTEMPTS: int = 0

    # Dedup + Relevance Scorer Settings (Phase 1)
    CHUNK_SIZE: int = 500  # words per chunk
    CHUNK_OVERLAP: int = 50  # word overlap between chunks
    SIMILARITY_THRESHOLD: float = 0.85  # cosine similarity threshold for dedup
    TOP_K_CHUNKS: int = 60  # Keep top 50-80 chunks (60 for 6-8 chapters)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local sentence-transformers model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for relevance
    MIN_CHUNK_WORDS: int = 200  # Minimum words in extracted text to keep chunk

    # Phase 2: Chapter Planner Settings
    CHAPTER_PLANNER_MODEL: str = "gpt-5.1"
    CHAPTER_PLANNER_TEMPERATURE: float = 0.7
    CHAPTER_PLANNER_BATCH_SIZE: int = 5
    MIN_CHAPTERS: int = 6
    MAX_CHAPTERS: int = 8
    TARGET_DURATION_MINUTES: float = 26.0
    MIN_CHAPTER_DURATION: float = 2.0
    MAX_CHAPTER_DURATION: float = 5.0
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.45

    # Phase 2: Character Designer Settings
    CHARACTER_DESIGNER_MODEL: str = "gpt-5.1"
    CHARACTER_DESIGNER_TEMPERATURE: float = 0.8

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)
