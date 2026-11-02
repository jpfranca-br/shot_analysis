from contextlib import asynccontextmanager
from pathlib import Path
import shutil
import threading
import uuid
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.analytics import compute_analytics
from backend.audio_engine import (
    detect_events_with_limit,
    detect_events_with_profile,
    extract_normalized_mono_pcm,
    update_signature_profile,
)
from backend.config import AppConfig, load_config
from backend.demo_generator import generate_synthetic_practice_video
from backend.models import (
    FirearmCreatePayload,
    FirearmUpdatePayload,
    SessionAnnotationPayload,
    SessionCheckPayload,
    SessionRecord,
)
from backend.multimodal_detector import evaluate_session_predictions
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
    storage_fn = app.dependency_overrides.get(get_storage, get_storage)
    storage = storage_fn()
    storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )
    yield
    cfg_exit = load_config()
    storage_exit_fn = app.dependency_overrides.get(get_storage, get_storage)
    storage_exit = storage_exit_fn()
    storage_exit.cleanup_videos(
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


@app.post("/api/firearms")
def create_firearm_endpoint(
    payload: FirearmCreatePayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    cleaned = payload.gun_name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="Firearm name cannot be empty.")
    sig = storage.create_firearm(
        gun_name=cleaned,
        beep_frequency_hz=payload.beep_frequency_hz,
        beep_duration_ms=payload.beep_duration_ms,
        shot_decay_rate=payload.shot_decay_rate,
    )
    return sig.model_dump()


@app.post("/api/firearms/restore-defaults")
def restore_default_firearms_endpoint(
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    signatures = storage.restore_default_firearms()
    return {
        "firearms": [sig.model_dump() for sig in signatures],
    }


@app.get("/api/firearms/{gun_name}")
def get_firearm_signature(
    gun_name: str, storage: StorageManager = Depends(get_storage)
) -> Dict[str, Any]:
    sig = storage.get_signature(gun_name, include_untrained=True)
    if not sig:
        raise HTTPException(status_code=404, detail="Firearm signature not found")
    return sig.model_dump()


@app.put("/api/firearms/{gun_name}")
def update_firearm_endpoint(
    gun_name: str,
    payload: FirearmUpdatePayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    updated = storage.update_firearm(
        gun_name=gun_name,
        new_gun_name=payload.new_gun_name,
        beep_frequency_hz=payload.beep_frequency_hz,
        beep_duration_ms=payload.beep_duration_ms,
        shot_decay_rate=payload.shot_decay_rate,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Firearm not found")
    return updated.model_dump()


@app.post("/api/firearms/{gun_name}/reset")
def reset_firearm_signature_endpoint(
    gun_name: str,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    reset_sig = storage.reset_signature(gun_name)
    if not reset_sig:
        raise HTTPException(status_code=404, detail="Firearm not found")
    return reset_sig.model_dump()


@app.delete("/api/firearms/{gun_name}")
def delete_firearm_endpoint(
    gun_name: str,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    deleted = storage.delete_signature(gun_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Firearm not found")
    return {"deleted": True, "gun_name": gun_name}


def _finalize_uploaded_video_session(
    video_id: str,
    orig_filename: str,
    ext: str,
    cleaned_gun: str,
    storage: StorageManager,
) -> Dict[str, Any]:
    saved_video_name = f"{video_id}{ext}"
    saved_wav_name = f"{video_id}.wav"
    video_path = storage.uploads_dir / saved_video_name
    wav_path = storage.uploads_dir / saved_wav_name

    if not video_path.exists() or video_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty or missing.")

    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(
        f"[Upload {video_id}] Saved {size_mb:.2f} MB ('{orig_filename}'). Extracting mono PCM audio via FFmpeg...",
        flush=True,
    )

    # FR-1.3: Extract audio track into normalized mono PCM stream
    duration_sec = extract_normalized_mono_pcm(video_path, wav_path)
    print(
        f"[Upload {video_id}] Audio extracted ({duration_sec:.2f}s). Checking signature for '{cleaned_gun}'...",
        flush=True,
    )

    # FR-2.1: Check local database for existing audio profile associated with firearm name
    signature = storage.get_signature(cleaned_gun)

    if signature is None:
        # FR-2.2: Training Mode (Cold Start)
        mode = "training"
        beep_timestamp = None
        end_beep_timestamp = None
        shot_timestamps = []
        sig_version = None
        print(f"[Upload {video_id}] Cold Start (training mode) ready.", flush=True)
    else:
        # FR-2.3: Automated Mode using Multimodal (Video + Audio) detection
        mode = "automated"
        print(
            f"[Upload {video_id}] Running multimodal detection (v{signature.signature_version})...",
            flush=True,
        )
        beep_timestamp, shot_timestamps, end_beep_timestamp = detect_events_with_limit(
            wav_path=wav_path,
            signature=signature,
            video_path=video_path,
            base_dir=storage.base_dir,
        )
        sig_version = signature.signature_version
        print(
            f"[Upload {video_id}] Detection complete: beep={beep_timestamp}, limit={end_beep_timestamp}, shots={len(shot_timestamps)}",
            flush=True,
        )

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
        end_beep_timestamp=end_beep_timestamp,
        shot_timestamps=shot_timestamps,
        predicted_beep=beep_timestamp,
        predicted_end_beep=end_beep_timestamp,
        predicted_shots=list(shot_timestamps),
        user_modified=False,
    )
    storage.save_session(session)

    analytics = compute_analytics(
        beep_timestamp, shot_timestamps, end_beep_timestamp=end_beep_timestamp
    )

    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": signature.model_dump() if signature else None,
    }


_UPLOAD_WRITE_LOCK = threading.Lock()


@app.post("/api/upload/init")
def init_chunked_upload(
    filename: str = Query("practice.mp4"),
    gun_name: str = Query(...),
    total_bytes: int = Query(0),
    storage: StorageManager = Depends(get_storage),
    cfg: AppConfig = Depends(get_app_config),
) -> Dict[str, Any]:
    cleaned_gun = gun_name.strip()
    if not cleaned_gun:
        raise HTTPException(status_code=400, detail="Firearm name is required (FR-1.2).")

    orig_filename = filename or "practice.mp4"
    ext = Path(orig_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: .mp4, .mov, .webm (FR-1.1).",
        )

    storage.cleanup_videos(
        keep_uploaded=cfg.keep_uploaded,
        keep_rendered=cfg.keep_rendered,
    )

    video_id = uuid.uuid4().hex[:12]
    video_path = storage.uploads_dir / f"{video_id}{ext}"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with video_path.open("wb") as f:
        if total_bytes > 0:
            f.truncate(total_bytes)
    print(
        f"[Upload {video_id}] Initialized parallel chunked upload for '{orig_filename}' ({total_bytes / (1024 * 1024):.2f} MB, {cleaned_gun})",
        flush=True,
    )

    return {
        "video_id": video_id,
        "ext": ext,
        "filename": orig_filename,
        "gun_name": cleaned_gun,
        "total_bytes": total_bytes,
    }


@app.post("/api/upload/chunk/{video_id}")
async def append_upload_chunk(
    video_id: str,
    request: Request,
    ext: str = Query(".mp4"),
    offset: Optional[int] = Query(None),
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    safe_ext = ext.lower() if ext.lower() in ALLOWED_EXTENSIONS else ".mp4"
    video_path = storage.uploads_dir / f"{video_id}{safe_ext}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Upload session not initialized.")

    # Receive chunk over the network asynchronously (runs in parallel across all 8 connections)
    chunk = await request.body()
    if chunk:
        with _UPLOAD_WRITE_LOCK:
            if offset is not None:
                with video_path.open("r+b") as f:
                    f.seek(offset)
                    f.write(chunk)
            else:
                with video_path.open("ab") as f:
                    f.write(chunk)

    return {
        "video_id": video_id,
        "received_bytes": len(chunk),
        "offset": offset,
    }


@app.post("/api/upload/complete/{video_id}")
def complete_chunked_upload(
    video_id: str,
    filename: str = Query("practice.mp4"),
    gun_name: str = Query(...),
    ext: str = Query(".mp4"),
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    cleaned_gun = gun_name.strip()
    if not cleaned_gun:
        raise HTTPException(status_code=400, detail="Firearm name is required (FR-1.2).")
    safe_ext = ext.lower() if ext.lower() in ALLOWED_EXTENSIONS else ".mp4"
    return _finalize_uploaded_video_session(
        video_id=video_id,
        orig_filename=filename or "practice.mp4",
        ext=safe_ext,
        cleaned_gun=cleaned_gun,
        storage=storage,
    )


@app.post("/api/upload")
def upload_video(
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
    video_path = storage.uploads_dir / saved_video_name

    print(f"[Upload {video_id}] Saving uploaded file '{orig_filename}'...", flush=True)
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    return _finalize_uploaded_video_session(
        video_id=video_id,
        orig_filename=orig_filename,
        ext=ext,
        cleaned_gun=cleaned_gun,
        storage=storage,
    )


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
        end_beep_timestamp = None
        shot_timestamps = []
        sig_version = None
    else:
        mode = "automated"
        beep_timestamp, shot_timestamps, end_beep_timestamp = detect_events_with_limit(
            wav_path=wav_path,
            signature=signature,
            video_path=video_path,
            base_dir=storage.base_dir,
        )
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
        end_beep_timestamp=end_beep_timestamp,
        shot_timestamps=shot_timestamps,
        predicted_beep=beep_timestamp,
        predicted_end_beep=end_beep_timestamp,
        predicted_shots=list(shot_timestamps),
        user_modified=False,
    )
    storage.save_session(session)
    analytics = compute_analytics(
        beep_timestamp, shot_timestamps, end_beep_timestamp=end_beep_timestamp
    )

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
    analytics = compute_analytics(
        session.beep_timestamp,
        session.shot_timestamps,
        end_beep_timestamp=session.end_beep_timestamp,
    )
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
    end_beep_t = (
        round(float(payload.end_beep_timestamp), 4)
        if payload.end_beep_timestamp is not None
        else None
    )
    if beep_t is not None and end_beep_t is not None and end_beep_t < beep_t:
        beep_t, end_beep_t = end_beep_t, beep_t

    wav_path = storage.uploads_dir / f"{session.video_id}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Extracted session WAV file not found")

    # Locate uploaded video file for multimodal (Video + Audio) training
    input_video_path = None
    for cand in storage.uploads_dir.glob(f"{session.video_id}.*"):
        if cand.suffix.lower() in ALLOWED_EXTENSIONS:
            input_video_path = cand
            break

    existing_sig = storage.get_signature(gun_name)
    updated_sig = update_signature_profile(
        existing=existing_sig,
        gun_name=gun_name,
        wav_path=wav_path,
        beep_timestamp=beep_t,
        shot_timestamps=sorted_shots,
        video_path=input_video_path,
        video_id=session.video_id,
        base_dir=storage.base_dir,
    )
    storage.save_signature(updated_sig)

    session.gun_name = gun_name
    session.beep_timestamp = beep_t
    session.end_beep_timestamp = end_beep_t
    session.shot_timestamps = sorted_shots
    session.user_modified = payload.user_modified
    session.signature_version = updated_sig.signature_version
    storage.save_session(session)

    analytics = compute_analytics(
        beep_t, sorted_shots, end_beep_timestamp=end_beep_t
    )

    return {
        "session": session.model_dump(),
        "analytics": analytics.model_dump(),
        "signature": updated_sig.model_dump(),
    }


@app.post("/api/check")
def check_model_predictions_endpoint(
    payload: SessionCheckPayload,
    storage: StorageManager = Depends(get_storage),
) -> Dict[str, Any]:
    """
    Evaluate automated multimodal model predictions on a Check video against user-verified
    ground truth timestamps without modifying the trained model.
    """
    session = storage.get_session(payload.video_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    gun_name = payload.gun_name.strip() or session.gun_name
    signature = storage.get_signature(gun_name)

    pred_beep = payload.predicted_beep
    pred_shots = payload.predicted_shots

    if pred_beep is None and pred_shots is None:
        if session.predicted_beep is not None or session.predicted_shots:
            pred_beep = session.predicted_beep
            pred_shots = session.predicted_shots
        elif signature is not None:
            wav_path = storage.uploads_dir / f"{session.video_id}.wav"
            input_video_path = None
            for cand in storage.uploads_dir.glob(f"{session.video_id}.*"):
                if cand.suffix.lower() in ALLOWED_EXTENSIONS:
                    input_video_path = cand
                    break
            if wav_path.exists():
                pred_beep, pred_shots = detect_events_with_profile(
                    wav_path=wav_path,
                    signature=signature,
                    video_path=input_video_path,
                    base_dir=storage.base_dir,
                )

    result = evaluate_session_predictions(
        gun_name=gun_name,
        signature_version=signature.signature_version if signature else 0,
        ground_truth_beep=payload.ground_truth_beep,
        ground_truth_shots=payload.ground_truth_shots,
        predicted_beep=pred_beep,
        predicted_shots=pred_shots or [],
    )
    return {
        "check_result": result.model_dump(),
        "predicted_beep": pred_beep,
        "predicted_shots": pred_shots or [],
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
        end_beep_timestamp=updated_session.end_beep_timestamp,
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


@app.get("/api/videos/{video_id}/info")
def get_rendered_video_info(
    video_id: str, storage: StorageManager = Depends(get_storage)
) -> Dict[str, Any]:
    session = storage.get_session(video_id)
    output_path = storage.outputs_dir / f"{video_id}_annotated.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found")

    safe_gun = (
        session.gun_name.replace(" ", "_").lower() if session else "practice"
    )
    download_name = f"{safe_gun}_{video_id}_analyzed.mp4"
    return {
        "video_id": video_id,
        "size_bytes": output_path.stat().st_size,
        "filename": download_name,
    }


@app.get("/api/videos/{video_id}/chunk")
def download_rendered_video_chunk(
    video_id: str,
    start: int = Query(0, ge=0),
    end: int = Query(..., gt=0),
    storage: StorageManager = Depends(get_storage),
) -> Response:
    output_path = storage.outputs_dir / f"{video_id}_annotated.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Rendered video not found")

    file_size = output_path.stat().st_size
    safe_start = min(start, file_size)
    safe_end = min(end, file_size)
    length = max(0, safe_end - safe_start)

    with output_path.open("rb") as f:
        f.seek(safe_start)
        data = f.read(length)

    return Response(content=data, media_type="application/octet-stream")


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
