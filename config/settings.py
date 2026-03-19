"""Application configuration settings."""
from pydantic_settings import BaseSettings
from pathlib import Path

# Tier identifiers — per-phase settings reference these
TIER_LOW: str = "low"
TIER_MEDIUM: str = "medium"
TIER_HIGH: str = "high"
TARGET_LANGUAGE:str = "hi-IN"

# Per-model pricing (USD per million tokens) — update when providers change rates
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI LLM models
    "gpt-5.4-nano": {"input_per_million": 0.02, "output_per_million": 1.25},
    "gpt-4.1-mini":  {"input_per_million": 0.40, "output_per_million": 1.60},
    "gpt-5.4-mini":  {"input_per_million": 0.75, "output_per_million": 4.50},
    "gpt-4o-mini":   {"input_per_million": 0.15, "output_per_million": 0.60},
    # Anthropic
    "claude-3-5-haiku-20241022": {"input_per_million": 0.80, "output_per_million": 4.00},
    "claude-sonnet-4-20250514":  {"input_per_million": 3.00, "output_per_million": 15.00},
    "claude-opus-4-20250514":    {"input_per_million": 15.00, "output_per_million": 75.00},
    # Sarvam AI LLM
    "sarvam-30b":  {"input_per_million": 0.08, "output_per_million": 0.30},
    "sarvam-105b": {"input_per_million": 0.19, "output_per_million": 0.72},
    # Gemini LLM
    "gemini-2.0-flash": {"input_per_million": 0.10, "output_per_million": 0.40},
    "gemini-2.5-flash": {"input_per_million": 0.15, "output_per_million": 0.60},
    "gemini-2.5-pro":   {"input_per_million": 1.25, "output_per_million": 10.00},
    # Vertex AI Gemini TTS models
    "gemini-2.5-pro-preview-tts": {"input_per_million": 75.0, "output_per_million": 0.0},
    "gemini-2.5-pro-tts":         {"input_per_million": 3.50, "output_per_million": 0.0},
    "gemini-2.5-flash-tts":       {"input_per_million": 0.10, "output_per_million": 0.0},
    # OpenAI TTS
    "gpt-4o-mini-tts": {"input_per_million": 12.00, "output_per_million": 0.0},
    # Sarvam TTS
    "bulbul:v2":       {"input_per_million": 0.10, "output_per_million": 0.0},
}

# Currency conversion
USD_TO_INR: float = 95.0


class Settings(BaseSettings):
    """Application settings."""

    # LLM Provider: "openai", "anthropic", "sarvam", "gemini"
    LLM_PROVIDER: str = "openai"

    # Per-provider model tiers (LOW / MEDIUM / HIGH)
    OPENAI_MODEL_LOW: str = "gpt-5.4-nano"
    OPENAI_MODEL_MEDIUM: str = "gpt-4.1-mini"
    OPENAI_MODEL_HIGH: str = "gpt-5.4-mini"

    ANTHROPIC_MODEL_LOW: str = "claude-3-5-haiku-20241022"
    ANTHROPIC_MODEL_MEDIUM: str = "claude-sonnet-4-20250514"
    ANTHROPIC_MODEL_HIGH: str = "claude-opus-4-20250514"

    SARVAM_MODEL_LOW: str = "sarvam-105b"
    SARVAM_MODEL_MEDIUM: str = "sarvam-105b"
    SARVAM_MODEL_HIGH: str = "sarvam-105b"

    GEMINI_MODEL_LOW: str = "gemini-2.0-flash"
    GEMINI_MODEL_MEDIUM: str = "gemini-2.5-flash"
    GEMINI_MODEL_HIGH: str = "gemini-2.5-pro"

    # Phase-level tier assignments
    QUERY_PRODUCER_MODEL: str = TIER_LOW
    QUERY_PRODUCER_TEMPERATURE: float = 1.0

    # API Settings
    API_TITLE: str = "AI Podcast Generator"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Automated Multi-Speaker Podcast Generation Pipeline"

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    GCP_PROJECT_ID: str = "effortless-lock-329115"
    GCP_LOCATION: str = "us-central1"

    # Search provider: "google" or "tavily"
    SEARCH_PROVIDER: str = "tavily" #google

    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = DATA_DIR / "cache"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    INPUT_DIR: Path = DATA_DIR / "input"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    TEMP_DIR: Path = DATA_DIR / "temp"

    # Processing Settings
    MIN_PODCAST_DURATION_SEC: int = 900  # 15 minutes
    MAX_PODCAST_DURATION_SEC: int = 1800  # 30 minutes
    NUM_SPEAKERS: int = 3
    PHASE4_SYNTHESIS_MINUTES_CAP: float = 15.0  # Max minutes of dialogue to synthesise per run

    #Google Web search settings
    SEARCH_RESULTS_PER_QUERY: int = 5

    #query Producer settings
    QUERY_GENERATION_TEMPERATURE: float = 0.7
    QUERY_PRODUCE_PER_TOPIC: int = 15

    # Web Scraper Settings
    WEB_SCRAPER_MAX_WORKERS: int = 50
    WEB_SCRAPER_TIMEOUT: int = 15  # seconds per request
    LINK_FAILURE_THRESHOLD: float = 0.3  # 30% failure triggers query rewrite
    MAX_QUERY_REWRITE_ATTEMPTS: int = 1

    # Dedup + Relevance Scorer Settings (Phase 1)
    CHUNK_SIZE: int = 500  # words per chunk
    CHUNK_OVERLAP: int = 50  # word overlap between chunks
    SIMILARITY_THRESHOLD: float = 0.85  # cosine similarity threshold for dedup
    TOP_K_CHUNKS: int = 60  # Keep top 50-80 chunks (60 for 6-8 chapters)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local sentence-transformers model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder for relevance
    MIN_CHUNK_WORDS: int = 200  # Minimum words in extracted text to keep chunk

    # Phase 2: Chapter Planner Settings
    CHAPTER_PLANNER_MODEL: str = TIER_LOW
    CHAPTER_PLANNER_TEMPERATURE: float = 1.0
    CHAPTER_PLANNER_BATCH_SIZE: int = 2
    MIN_CHAPTERS: int = 6
    MAX_CHAPTERS: int = 8
    TARGET_DURATION_MINUTES: float = 26.0
    MIN_CHAPTER_DURATION: float = 2.0
    MAX_CHAPTER_DURATION: float = 5.0
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.45

    # Phase 2: Character Designer Settings
    CHARACTER_DESIGNER_MODEL: str = TIER_HIGH
    CHARACTER_DESIGNER_TEMPERATURE: float = 1.0

    # Phase 3: Dialogue Generation Settings
    DIALOGUE_ENGINE_MODEL: str = TIER_HIGH
    DIALOGUE_ENGINE_TEMPERATURE: float = 1.0
    EXPERT_EXPANDER_MODEL: str = TIER_LOW
    EXPERT_EXPANDER_TEMPERATURE: float = 1.0
    NATURALNESS_MODEL: str = TIER_LOW
    NATURALNESS_TEMPERATURE: float = 1.0
    FACT_CHECKER_MODEL: str = TIER_LOW
    FACT_CHECKER_TEMPERATURE: float = 1.0
    QA_REVIEWER_MODEL: str = TIER_LOW
    QA_REVIEWER_TEMPERATURE: float = 1.0
    PHASE3_ENABLE_FACT_CHECKER: bool = False
    PHASE3_ENABLE_QA_REVIEWER: bool = False
    PHASE3_MAX_RETRIES: int = 2

    # TTS Provider: "google", "openai", "elevenlabs", or "sarvam" — switch on the fly
    TTS_PROVIDER: str = 'sarvam'#'openai' #"google"
    TTS_FALLBACK_PROVIDER: str = ""  # empty = no fallback; set to "openai", "elevenlabs", etc. to enable
    GOOGLE_TTS_MODEL: str = "gemini-2.5-pro-preview-tts"
    ELEVENLABS_TTS_MODEL: str = "eleven_v3"
    OPENAI_TTS_MODEL: str = "gpt-4o-mini-tts"
    SARVAM_TTS_MODEL: str = "bulbul:v3"
    PHASE4_RAW_AUDIO_DIR: Path = AUDIO_DIR / "raw"
    PHASE4_MAX_WORKERS: int = 6
    PHASE4_MAX_CONCURRENT_API_CALLS: int = 6  # semaphore cap on simultaneous Gemini calls
    PHASE4_MAX_RETRIES: int = 3
    PHASE4_REQUEST_TIMEOUT_SECONDS: int = 60
    PHASE4_RETRY_BASE_SECONDS: float = 10.0
    PHASE4_MIN_REQUEST_GAP_SECONDS: float = 3.0  # kept for reference; no longer used for gating
    PHASE4_MAX_TEXT_CHARS_PER_JOB: int = 1200
    PHASE4_DEFAULT_TURN_GAP_SECONDS: float = 0.2
    PHASE4_MIN_DURATION_SECONDS: float = 0.15
    PHASE4_SILENCE_PEAK_THRESHOLD: int = 64
    PHASE4_CLIPPING_SAMPLE_THRESHOLD: int = 32760
    PHASE4_CLIPPING_RATIO_THRESHOLD: float = 0.01
    PHASE4_TARGET_SAMPLE_RATE: int = 22050  # Sarvam Bulbul v3 outputs 22050 Hz WAV
    PHASE4_TARGET_CHANNELS: int = 1
    PHASE4_MAX_FAILURE_RATIO: float = 0.02  # fraction of clips allowed to fail before blocking Phase 5 (0.0 = strict)
    GOOGLE_TTS_ALLOWED_VOICES: tuple[str, ...] = (
        "Aoede",
        "Charon",
        "Puck",
        "Kore",
        "Fenrir",
        "Leda",
        "Enceladus",
        "Vindemiatrix",
        "Zubenelgenubi",
        "Zephyr",
    )
    GOOGLE_TTS_HOST_VOICE: str = "Aoede"
    GOOGLE_TTS_EXPERT_VOICE: str = "Charon"
    GOOGLE_TTS_SKEPTIC_VOICE: str = "Kore"
    ELEVENLABS_HOST_VOICE_ID: str = ""
    ELEVENLABS_EXPERT_VOICE_ID: str = ""
    ELEVENLABS_SKEPTIC_VOICE_ID: str = ""
    OPENAI_TTS_ALLOWED_VOICES: tuple[str, ...] = (
        "alloy", "ash", "ballad", "cedar", "coral", "echo",
        "fable", "marin", "nova", "onyx", "sage", "shimmer",
    )
    OPENAI_TTS_HOST_VOICE: str = "coral"
    OPENAI_TTS_EXPERT_VOICE: str = "onyx"
    OPENAI_TTS_SKEPTIC_VOICE: str = "nova"
    SARVAM_TTS_HOST_VOICE: str = "tanya"      # was "meera" — removed from API
    SARVAM_TTS_EXPERT_VOICE: str = "advait"   # was "arvind" — removed from API
    SARVAM_TTS_SKEPTIC_VOICE: str = "ritu"    # still valid
    SARVAM_TTS_MAX_CHARS_PER_JOB: int = 490   # Sarvam Bulbul API limit is 500 chars/input

    # Phase 5: Audio Post-Processing
    PHASE5_TARGET_SAMPLE_RATE: int = 44100
    PHASE5_TARGET_CHANNELS: int = 2
    PHASE5_TARGET_SAMPLE_WIDTH: int = 2

    # Overlap Engine
    PHASE5_TURN_GAP_MS: int = 300
    PHASE5_CROSSFADE_MS: int = 75
    PHASE5_INTERRUPT_VOLUME_REDUCTION_DB: int = -3
    PHASE5_BACKCHANNEL_VOLUME_DB: int = -8
    PHASE5_LAUGH_VOLUME_DB: int = -4
    PHASE5_INTERRUPT_MAX_DUAL_PLAY_MS: int = 3000  # max simultaneous playback
    PHASE5_INTERRUPT_FADE_OUT_MS: int = 500  # fade-out for interrupted speaker

    # Post-Processor (EQ)
    PHASE5_EQ_PRESENCE_FREQ: int = 3000
    PHASE5_EQ_PRESENCE_GAIN: int = 2
    PHASE5_EQ_RUMBLE_FREQ: int = 80
    PHASE5_EQ_RUMBLE_GAIN: int = -6

    # Post-Processor (Compression)
    PHASE5_COMP_THRESHOLD_DB: int = -20
    PHASE5_COMP_RATIO: int = 2
    PHASE5_COMP_ATTACK_MS: int = 5
    PHASE5_COMP_RELEASE_MS: int = 50
    PHASE5_COMP_MAKEUP_GAIN_DB: int = 2

    # Post-Processor (Loudness)
    PHASE5_LOUDNESS_TARGET_LUFS: float = -16.0
    PHASE5_LOUDNESS_TRUE_PEAK_DB: float = -1.5
    PHASE5_NOISE_GATE_THRESHOLD_DB: int = -40
    PHASE5_NOISE_GATE_SILENCE_DURATION: float = 0.1

    # Post-Processor (Room Tone)
    PHASE5_ENABLE_ROOM_TONE: bool = False
    PHASE5_ROOM_TONE_LEVEL_DB: int = -32

    # Cold Open
    PHASE5_COLD_OPEN_MIN_MS: int = 12000
    PHASE5_COLD_OPEN_MAX_MS: int = 25000
    PHASE5_COLD_OPEN_LLM_MODEL: str = TIER_MEDIUM

    # Chapter Stitcher
    PHASE5_INTRO_MUSIC_DURATION_MS: int = 8000
    PHASE5_COLD_OPEN_INTRO_CROSSFADE_MS: int = 3000
    PHASE5_MP3_BITRATE_KBPS: int = 128

    # Output
    PHASE5_OUTPUT_BASE_DIR: str = "data/audio/phase5"

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()

# Ensure directories exist
for _dir in (
    settings.DATA_DIR, settings.CACHE_DIR, settings.AUDIO_DIR,
    settings.INPUT_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR,
    settings.PHASE4_RAW_AUDIO_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
