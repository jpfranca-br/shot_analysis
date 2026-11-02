from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import uuid
from typing import Any, Dict

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.analytics import compute_analytics
from backend.audio_engine import (
    detect_events_with_profile,
    extract_normalized_mono_pcm,
    update_signature_profile,
)
from backend.config import AppConfig, load_config
from backend.demo_generator import generate_synthetic_practice_video
from backend.models import SessionAnnotationPayload, SessionRecord
from backend.storage import StorageManager
from backend.video_engine import render_annotated_video

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".webm"}


def get_storage() -> StorageManager:
    return StorageManager()


def get_app_config() -> AppConfig:
    return load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    storage = StorageManager()
    storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )
    yield
    cfg_exit = load_config()
    storage.cleanup_videos(
        keep_uploaded=cfg_exit.keep_uploaded,
        keep_rendered=cfg_exit.keep_rendered,
    )


app = FastAPI(
    title="Shooting Practice Video Analyzer",
    description="Audio signature learning, interactive waveform annotation, split timing, and OpenCV video overlay analyzer.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config_endpoint(
    cfg: AppConfig = Depends(get_app_config),
) -> Dict[str, Any]:
    return {
        "store_videos": cfg.store_videos,
        "store_uploaded_videos": cfg.keep_uploaded,
        "store_rendered_videos": cfg.keep_rendered,
    }


@app.post("/api/cleanup")
def cleanup_ephemeral_videos(
    storage: StorageManager = Depends(get_storage),
    cfg: AppConfig = Depends(get_app_config),
) -> Dict[str, Any]:
    deleted = storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )
    return {
        "deleted_files": deleted,
        "keep_uploaded": cfg.keep_uploaded,
        "keep_rendered": cfg.keep_rendered,
    }


@app.get("/api/firearms")
def list_firearms(storage: StorageManager = Depends(get_storage)) -> Dict[str, Any]:
    signatures = storage.list_signatures()
    return {
        "firearms": [sig.model_dump() for sig in signatures],
    }


@app.get("/api/firearms/{gun_name}")
def get_firearm_signature(
    gun_name: str, storage: StorageManager = Depends(get_storage)
) -> Dict[str, Any]:
    sig = storage.get_signature(gun_name)
    if not sig:
        raise HTTPException(status_code=404, detail="Firearm signature not found")
    return sig.model_dump()


@app.post("/api/upload")
async def upload_video(
    video: UploadFile = File(...),
    gun_name: str = Form(...),
    storage: StorageManager = Depends(get_storage),
    cfg: AppConfig = Depends(get_app_config),
) -> Dict[str, Any]:
    cleaned_gun = gun_name.strip()
    if not cleaned_gun:
        raise HTTPException(status_code=400, detail="Firearm name is required (FR-1.2).")

    orig_filename = video.filename or "practice.mp4"
    ext = Path(orig_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: .mp4, .mov, .webm (FR-1.1).",
        )

    # Purge previous sessions' videos if config.json specifies not to store videos
    storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )

    video_id = uuid.uuid4().hex[:12]
    saved_video_name = f"{video_id}{ext}"
    saved_wav_name = f"{video_id}.wav"

    video_path = storage.uploads_dir / saved_video_name
    wav_path = storage.uploads_dir / saved_wav_name

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # FR-1.3: Extract audio track into normalized mono PCM stream
    duration_sec = extract_normalized_mono_pcm(video_path, wav_path)

    # FR-2.1: Check local database for existing audio profile associated with firearm name
    signature = storage.get_signature(cleaned_gun)

    if signature is None:
        # FR-2.2: Training Mode (Cold Start)
        mode = "training"
        beep_timestamp = None
        shot_timestamps = []
        sig_version = None
    else:
        # FR-2.3: Automated Mode using librosa peak detection & template matching
        mode = "automated"
        beep_timestamp, shot_timestamps = detect_events_with_profile(wav_path, signature)
        sig_version = signature.signature_version

    session = SessionRecord(
        video_id=video_id,
        filename=orig_filename,
        gun_name=cleaned_gun,
        video_url=f"/api/media/uploads/{saved_video_name}",
        audio_url=f"/api/media/uploads/{saved_wav_name}",
        duration_sec=duration_sec,
        mode=mode,
        signature_version=sig_version,
        beep_timestamp=beep_timestamp,
        shot_timestamps=shot_timestamps,
        user_modified=False,
    )
    storage.save_session(session)

    analytics = compute_analytics(beep_timestamp, shot_timestamps)

    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": signature.model_dump() if signature else None,
    }


@app.post("/api/demo-upload")
def create_demo_session(
    gun_name: str = Form("Glock 19"),
    storage: StorageManager = Depends(get_storage),
    cfg: AppConfig = Depends(get_app_config),
) -> Dict[str, Any]:
    """Generate and ingest a sample shooting practice video for rapid demonstration."""
    storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )
    cleaned_gun = gun_name.strip() or "Glock 19"
    video_id = uuid.uuid4().hex[:12]
    saved_video_name = f"{video_id}.mp4"
    saved_wav_name = f"{video_id}.wav"

    video_path = storage.uploads_dir / saved_video_name
    wav_path = storage.uploads_dir / saved_wav_name

    generate_synthetic_practice_video(video_path)
    duration_sec = extract_normalized_mono_pcm(video_path, wav_path)
    signature = storage.get_signature(cleaned_gun)

    if signature is None:
        mode = "training"
        beep_timestamp = None
        shot_timestamps = []
        sig_version = None
    else:
        mode = "automated"
        beep_timestamp, shot_timestamps = detect_events_with_profile(wav_path, signature)
        sig_version = signature.signature_version

    session = SessionRecord(
        video_id=video_id,
        filename="sample_range_drill.mp4",
        gun_name=cleaned_gun,
        video_url=f"/api/media/uploads/{saved_video_name}",
        audio_url=f"/api/media/uploads/{saved_wav_name}",
        duration_sec=duration_sec,
        mode=mode,
        signature_version=sig_version,
        beep_timestamp=beep_timestamp,
        shot_timestamps=shot_timestamps,
        user_modified=False,
    )
    storage.save_session(session)
    analytics = compute_analytics(beep_timestamp, shot_timestamps)

    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": signature.model_dump() if signature else None,
    }


@app.get("/api/sessions/{video_id}")
def get_session_details(
    video_id: str, storage: StorageManager = Depends(get_storage)
) -> Dict[str, Any]:
    session = storage.get_session(video_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    signature = storage.get_signature(session.gun_name)
    analytics = compute_analytics(session.beep_timestamp, session.shot_timestamps)
    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": signature.model_dump() if signature else None,
    }


@app.post("/api/annotations")
def save_and_refine_annotations(
    payload: SessionAnnotationPayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """
    Save confirmed or user-edited timestamps and refine the firearm's audio signature (FR-2.4).
    """
    session = storage.get_session(payload.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    gun_name = payload.gun_name.strip() or session.gun_name
    sorted_shots = sorted(round(float(t), 4) for t in payload.shot_timestamps if t is not None)
    beep_t = round(float(payload.beep_timestamp), 4) if payload.beep_timestamp is not None else None

    wav_path = storage.uploads_dir / f"{session.video_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Extracted session WAV file not found")

    existing_sig = storage.get_signature(gun_name)
    updated_sig = update_signature_profile(
        existing=existing_sig,
        gun_name=gun_name,
        wav_path=wav_path,
        beep_timestamp=beep_t,
        shot_timestamps=sorted_shots,
    )
    storage.save_signature(updated_sig)

    session.gun_name = gun_name
    session.beep_timestamp = beep_t
    session.shot_timestamps = sorted_shots
    session.user_modified = payload.user_modified
    session.signature_version = updated_sig.signature_version
    storage.save_session(session)

    analytics = compute_analytics(beep_t, sorted_shots)

    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": updated_sig.model_dump(),
    }


@app.post("/api/render")
def process_and_render_video(
    payload: SessionAnnotationPayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """
    Confirm annotations, refine profile (FR-2.4), and generate annotated video with burnt-in overlays (FR-5.1 - FR-5.3).
    """
    session = storage.get_session(payload.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # First save and refine the profile with the confirmed timestamps
    annotation_result = save_and_refine_annotations(payload, storage)
    updated_session = SessionRecord(**annotation_result["session"])

    # Locate uploaded video file
    input_candidates = list(storage.uploads_dir.glob(f"{session.video_id}.*"))
    input_video_path = None
    for cand in input_candidates:
        if cand.suffix.lower() in ALLOWED_EXTENSIONS:
            input_video_path = cand
            break

    if input_video_path is None or not input_video_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded source video file not found")

    output_filename = f"{session.video_id}_annotated.mp4"
    output_video_path = storage.outputs_dir / output_filename

    render_annotated_video(
        input_video_path=input_video_path,
        output_video_path=output_video_path,
        gun_name=updated_session.gun_name,
        beep_timestamp=updated_session.beep_timestamp,
        shot_timestamps=updated_session.shot_timestamps,
    )

    download_url = f"/api/videos/{session.video_id}/download"
    preview_url = f"/api/media/outputs/{output_filename}"

    updated_session.rendered_video_url = download_url
    storage.save_session(updated_session)

    return {
        "session": updated_session.model_dump(),
        "analytics": annotation_result["analytics"],
        "signature": annotation_result["signature"],
        "download_url": download_url,
        "preview_url": preview_url,
    }


@app.get("/api/media/uploads/{filename}")
def get_uploaded_media(
    filename: str, storage: StorageManager = Depends(get_storage)
) -> FileResponse:
    file_path = storage.uploads_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(path=str(file_path))


@app.get("/api/media/outputs/{filename}")
def get_output_media(
    filename: str, storage: StorageManager = Depends(get_storage)
) -> FileResponse:
    file_path = storage.outputs_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Output media file not found")
    return FileResponse(path=str(file_path), media_type="video/mp4")


@app.get("/api/videos/{video_id}/download")
def download_rendered_video(
    video_id: str, storage: StorageManager = Depends(get_storage)
) -> FileResponse:
    session = storage.get_session(video_id)
    output_path = storage.outputs_dir / f"{video_id}_annotated.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found")

    safe_gun = (
        session.gun_name.replace(" ", "_").lower() if session else "practice"
    )
    download_name = f"{safe_gun}_{video_id}_analyzed.mp4"
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=download_name,
    )


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
