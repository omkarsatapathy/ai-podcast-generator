"""Application configuration settings."""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings."""
    #LLM settings
    QUERY_PRODUCER_MODEL: str = "gpt-4o-mini"
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
    CACHE_DIR: Path = DATA_DIR / "cache"
    AUDIO_DIR: Path = DATA_DIR / "audio"
    INPUT_DIR: Path = DATA_DIR / "input"
    OUTPUT_DIR: Path = DATA_DIR / "output"
    TEMP_DIR: Path = DATA_DIR / "temp"

    # Processing Settings
    MIN_PODCAST_DURATION_SEC: int = 900  # 15 minutes
    MAX_PODCAST_DURATION_SEC: int = 1800  # 30 minutes
    NUM_SPEAKERS: int = 2
    PHASE4_SYNTHESIS_MINUTES_CAP: float = 15.0  # Max minutes of dialogue to synthesise per run

    #Google Web search settings
    SEARCH_RESULTS_PER_QUERY: int = 10

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
    CHAPTER_PLANNER_MODEL: str = "gpt-4o-mini"
    CHAPTER_PLANNER_TEMPERATURE: float = 0.7
    CHAPTER_PLANNER_BATCH_SIZE: int = 5
    MIN_CHAPTERS: int = 6
    MAX_CHAPTERS: int = 8
    TARGET_DURATION_MINUTES: float = 26.0
    MIN_CHAPTER_DURATION: float = 2.0
    MAX_CHAPTER_DURATION: float = 5.0
    CLUSTER_SIMILARITY_THRESHOLD: float = 0.45

    # Phase 2: Character Designer Settings
    CHARACTER_DESIGNER_MODEL: str = "gpt-5.4-mini"
    CHARACTER_DESIGNER_TEMPERATURE: float = 0.8

    # Phase 3: Dialogue Generation Settings
    DIALOGUE_ENGINE_MODEL: str = "gpt-5.4-nano"
    DIALOGUE_ENGINE_TEMPERATURE: float = 0.8
    EXPERT_EXPANDER_MODEL: str = "gpt-5.4-nano"
    EXPERT_EXPANDER_TEMPERATURE: float = 0.7
    NATURALNESS_MODEL: str = "gpt-5.4-nano"
    NATURALNESS_TEMPERATURE: float = 0.6
    FACT_CHECKER_MODEL: str = "gpt-4o-mini"
    FACT_CHECKER_TEMPERATURE: float = 0.1
    QA_REVIEWER_MODEL: str = "gpt-5.4-nano"
    QA_REVIEWER_TEMPERATURE: float = 0.3
    PHASE3_MAX_RETRIES: int = 2

    # TTS Provider: "google" or "elevenlabs" — switch on the fly
    TTS_PROVIDER: str = "google"
    TTS_FALLBACK_PROVIDER: str = ""
    ELEVENLABS_API_KEY: str = ""
    GOOGLE_TTS_MODEL: str = "gemini-2.5-pro-preview-tts"
    ELEVENLABS_TTS_MODEL: str = "eleven_v3"
    PHASE4_RAW_AUDIO_DIR: Path = AUDIO_DIR / "raw"
    PHASE4_MAX_WORKERS: int = 1
    PHASE4_MAX_RETRIES: int = 3
    PHASE4_REQUEST_TIMEOUT_SECONDS: int = 60
    PHASE4_RETRY_BASE_SECONDS: float = 10.0
    PHASE4_MIN_REQUEST_GAP_SECONDS: float = 3.0  # min seconds between API calls
    PHASE4_MAX_TEXT_CHARS_PER_JOB: int = 1200
    PHASE4_DEFAULT_TURN_GAP_SECONDS: float = 0.2
    PHASE4_MIN_DURATION_SECONDS: float = 0.15
    PHASE4_SILENCE_PEAK_THRESHOLD: int = 64
    PHASE4_CLIPPING_SAMPLE_THRESHOLD: int = 32760
    PHASE4_CLIPPING_RATIO_THRESHOLD: float = 0.01
    PHASE4_TARGET_SAMPLE_RATE: int = 24000
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
    PHASE5_COLD_OPEN_LLM_MODEL: str = "gpt-5.4-nano"

    # Chapter Stitcher
    PHASE5_INTRO_MUSIC_DURATION_MS: int = 8000
    PHASE5_COLD_OPEN_INTRO_CROSSFADE_MS: int = 3000
    PHASE5_MP3_BITRATE_KBPS: int = 128

    # Output
    PHASE5_OUTPUT_BASE_DIR: str = "data/audio/phase5"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Ensure directories exist
for _dir in (
    settings.DATA_DIR, settings.CACHE_DIR, settings.AUDIO_DIR,
    settings.INPUT_DIR, settings.OUTPUT_DIR, settings.TEMP_DIR,
    settings.PHASE4_RAW_AUDIO_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)
