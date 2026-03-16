"""FastAPI application for AI Podcast Generator - Stage 1."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict

from config.settings import settings
from src.models.schemas import (
    PodcastRequest,
    PodcastResponse,
    JobStatusResponse,
    PodcastStatus,
    HealthResponse
)
from src.utils.logger import logger
from src.utils.helpers import generate_job_id, get_timestamp_iso, get_output_path


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (for Stage 1 - will be replaced with DB later)
jobs_db: Dict[str, dict] = {}


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return HealthResponse(
        status="healthy",
        version=settings.API_VERSION
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.API_VERSION
    )


@app.post("/api/v1/generate", response_model=PodcastResponse)
async def generate_podcast(
    request: PodcastRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a podcast from a topic.

    This endpoint accepts a topic and optional description, then initiates
    the podcast generation pipeline. The process runs asynchronously.

    Args:
        request: PodcastRequest with topic and optional parameters
        background_tasks: FastAPI background tasks for async processing

    Returns:
        PodcastResponse with job_id and status
    """
    try:
        # Generate unique job ID
        job_id = generate_job_id()
        timestamp = get_timestamp_iso()

        logger.info(f"Received podcast generation request - Job ID: {job_id}")
        logger.info(f"Topic: {request.topic}")

        # Initialize job in database
        jobs_db[job_id] = {
            "job_id": job_id,
            "topic": request.topic,
            "description": request.description,
            "num_speakers": request.num_speakers,
            "status": PodcastStatus.PENDING,
            "created_at": timestamp,
            "progress_percent": 0,
            "audio_url": None,
            "error": None
        }

        # Add background task for podcast generation
        background_tasks.add_task(process_podcast_generation, job_id)

        return PodcastResponse(
            job_id=job_id,
            status=PodcastStatus.PENDING,
            message="Podcast generation initiated successfully",
            audio_url=None,
            duration_seconds=None,
            created_at=timestamp
        )

    except Exception as e:
        logger.error(f"Error initiating podcast generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a podcast generation job.

    Args:
        job_id: Unique job identifier

    Returns:
        JobStatusResponse with current job status
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs_db[job_id]

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        message=get_status_message(job["status"]),
        progress_percent=job.get("progress_percent"),
        audio_url=job.get("audio_url"),
        error=job.get("error")
    )


@app.get("/api/v1/download/{job_id}")
async def download_podcast(job_id: str):
    """
    Download the generated podcast audio file.

    Args:
        job_id: Unique job identifier

    Returns:
        FileResponse with MP3 audio file
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = jobs_db[job_id]

    if job["status"] != PodcastStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Podcast not ready. Current status: {job['status']}"
        )

    audio_path = get_output_path(job_id, settings.OUTPUT_DIR)

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=f"{job_id}.mp3"
    )


async def process_podcast_generation(job_id: str):
    """
    Background task to process podcast generation.

    This is a placeholder for Stage 1. In later stages, this will orchestrate
    the full 5-phase pipeline.

    Args:
        job_id: Unique job identifier
    """
    try:
        logger.info(f"Starting podcast generation for job {job_id}")

        # Update status to processing
        jobs_db[job_id]["status"] = PodcastStatus.PROCESSING
        jobs_db[job_id]["progress_percent"] = 10

        # Stage 1: Placeholder for full pipeline
        # TODO: Implement Phase 1-5 pipeline orchestration
        logger.info(f"[Stage 1] Pipeline implementation pending for job {job_id}")

        # For now, mark as completed with a note
        jobs_db[job_id]["status"] = PodcastStatus.PENDING
        jobs_db[job_id]["progress_percent"] = 10
        jobs_db[job_id]["error"] = "Full pipeline not yet implemented (Stage 1)"

        logger.info(f"Job {job_id} - Stage 1 setup complete")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs_db[job_id]["status"] = PodcastStatus.FAILED
        jobs_db[job_id]["error"] = str(e)


def get_status_message(status: PodcastStatus) -> str:
    """Get human-readable status message."""
    messages = {
        PodcastStatus.PENDING: "Job is queued and waiting to be processed",
        PodcastStatus.PROCESSING: "Podcast is being generated",
        PodcastStatus.COMPLETED: "Podcast generation completed successfully",
        PodcastStatus.FAILED: "Podcast generation failed"
    }
    return messages.get(status, "Unknown status")


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting {settings.API_TITLE} v{settings.API_VERSION}")
    logger.info(f"Server running on http://{settings.HOST}:{settings.PORT}")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )
