"""
FastAPI application for SAM-Audio Source Separation Pipeline

Provides endpoints for uploading audio, processing with SAM-Audio,
and downloading separated audio tracks (original, isolated, residual, cleaned).
"""
import asyncio
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging

from processor import get_processor, MAX_AUDIO_DURATION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SAM-Audio Source Separation API",
    description="Isolate specific sounds from audio using Meta's SAM-Audio model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for tasks
TASKS: Dict[str, Dict] = {}
UPLOAD_DIR = Path("/tmp/sam-audio/uploads")
RESULTS_DIR = Path("/tmp/sam-audio/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default prompt - isolate ALL human speech/voices
# This captures news anchors, reporters, interviewees, and any spoken content
DEFAULT_PROMPT = "Human speech, people talking, voices, conversation, and spoken words"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sam-audio-pipeline"}


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "SAM-Audio Source Separation",
        "version": "1.0.0",
        "model": os.getenv("SAM_MODEL", "facebook/sam-audio-small"),
        "max_duration_minutes": MAX_AUDIO_DURATION // 60
    }


@app.post("/api/v1/pipelines/sam-audio/separate")
async def start_separation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    prompt: str = Form(default=DEFAULT_PROMPT)
):
    """
    Upload audio file and start source separation
    
    Args:
        file: Audio file (WAV, MP3, etc.) - max 65 minutes
        prompt: Text description of sound to isolate/remove
        
    Returns:
        Task ID for tracking progress
    """
    # Validate file type
    allowed_extensions = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac'}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{task_id}{file_ext}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Create task result directory
    result_dir = RESULTS_DIR / task_id
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize task
    TASKS[task_id] = {
        "status": "pending",
        "progress": 0,
        "total": 100,
        "message": "Task queued",
        "prompt": prompt,
        "created_at": datetime.now().isoformat(),
        "input_file": str(upload_path),
        "result_dir": str(result_dir),
        "original_path": None,
        "isolated_path": None,
        "residual_path": None,
        "cleaned_path": None,
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(
        process_audio_task,
        task_id=task_id,
        input_file=upload_path,
        prompt=prompt,
        result_dir=result_dir
    )
    
    logger.info(f"Created task {task_id} for file {file.filename} with prompt: '{prompt}'")
    
    return {
        "task_id": task_id,
        "message": "Separation started",
        "status": "pending",
        "prompt": prompt
    }


@app.get("/api/v1/pipelines/sam-audio/status/{task_id}")
async def get_task_status(task_id: str):
    """
    Get status of separation task
    
    Args:
        task_id: Task ID from separate endpoint
        
    Returns:
        Task status, progress, and available downloads
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    
    response = {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "total": task["total"],
        "message": task["message"],
        "prompt": task["prompt"],
        "created_at": task["created_at"],
        "error": task["error"]
    }
    
    # Add download info if completed
    if task["status"] == "completed":
        response["downloads"] = {
            "original": f"/api/v1/pipelines/sam-audio/download/{task_id}/original",
            "isolated": f"/api/v1/pipelines/sam-audio/download/{task_id}/isolated",
            "residual": f"/api/v1/pipelines/sam-audio/download/{task_id}/residual"
        }
        # Add cleaned track if available
        if task.get("cleaned_path"):
            response["downloads"]["cleaned"] = f"/api/v1/pipelines/sam-audio/download/{task_id}/cleaned"
        response["completed_at"] = task.get("completed_at")
    
    return response


@app.get("/api/v1/pipelines/sam-audio/download/{task_id}/{track_type}")
async def download_track(task_id: str, track_type: str):
    """
    Download separated audio track
    
    Args:
        task_id: Task ID from separate endpoint
        track_type: One of "original", "isolated", "residual", "cleaned"
        
    Returns:
        Audio file (WAV format)
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = TASKS[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Current status: {task['status']}"
        )
    
    # Map track type to file path
    path_map = {
        "original": task["original_path"],
        "isolated": task["isolated_path"],
        "residual": task["residual_path"],
        "cleaned": task.get("cleaned_path")
    }
    
    if track_type not in path_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid track type. Must be one of: {list(path_map.keys())}"
        )
    
    file_path = path_map[track_type]
    
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"{track_type} file not found")
    
    return FileResponse(
        path=file_path,
        filename=f"{track_type}_{task_id}.wav",
        media_type="audio/wav"
    )


async def process_audio_task(
    task_id: str, 
    input_file: Path, 
    prompt: str,
    result_dir: Path
):
    """
    Background task to process audio with SAM-Audio
    
    Args:
        task_id: Task ID
        input_file: Path to uploaded audio file
        prompt: Text prompt for separation
        result_dir: Directory to save results
    """
    def update_progress(current: int, total: int, message: str):
        """Callback to update task progress"""
        TASKS[task_id]["progress"] = current
        TASKS[task_id]["total"] = total
        TASKS[task_id]["message"] = message
    
    try:
        # Update status
        TASKS[task_id]["status"] = "processing"
        TASKS[task_id]["message"] = "Initializing model..."
        
        # Get processor
        processor = get_processor()
        
        # Run separation with silence removal enabled
        original_path, isolated_path, residual_path, cleaned_path = await processor.separate(
            audio_path=input_file,
            prompt=prompt,
            progress_callback=update_progress,
            remove_silence=True
        )
        
        # Move results to permanent location
        final_original = result_dir / "original.wav"
        final_isolated = result_dir / "isolated.wav"
        final_residual = result_dir / "residual.wav"
        
        shutil.move(str(original_path), str(final_original))
        shutil.move(str(isolated_path), str(final_isolated))
        shutil.move(str(residual_path), str(final_residual))
        
        # Move cleaned file if it exists
        final_cleaned = None
        if cleaned_path and cleaned_path.exists():
            final_cleaned = result_dir / "cleaned.wav"
            shutil.move(str(cleaned_path), str(final_cleaned))
        
        # Clean up temp directory
        original_path.parent.rmdir()
        
        # Update task
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["progress"] = 100
        TASKS[task_id]["message"] = "Separation complete!"
        TASKS[task_id]["original_path"] = str(final_original)
        TASKS[task_id]["isolated_path"] = str(final_isolated)
        TASKS[task_id]["residual_path"] = str(final_residual)
        TASKS[task_id]["cleaned_path"] = str(final_cleaned) if final_cleaned else None
        TASKS[task_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["message"] = f"Error: {str(e)}"
        TASKS[task_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)
