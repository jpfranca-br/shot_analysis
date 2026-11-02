from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

import cv2
import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf

from backend.models import BeepProfile, FirearmAudioSignature, ShotProfile
from backend.multimodal_detector import (
    SLOW_SHOT_THRESHOLD_SEC,
    cosine_similarity,
    detect_multimodal_events,
    detect_multimodal_events_with_limit,
    extract_cf032ba_window_features,
    extract_video_window_features,
    harvest_and_save_training_samples,
)

TARGET_SR = 22050


def extract_normalized_mono_pcm(video_path: Path, output_wav_path: Path) -> float:
    """
    Extract audio track from video into a normalized mono 16-bit PCM WAV file (FR-1.3).
    Returns the audio duration in seconds.
    """
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    temp_wav = output_wav_path.with_suffix(".raw.wav")
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-acodec",
        "pcm_s16le",
        str(temp_wav),
    ]

    result = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0 or not temp_wav.exists():
        duration = _get_video_duration_fallback(video_path)
        samples = np.zeros(int(max(duration, 1.0) * TARGET_SR), dtype=np.float32)
        sf.write(str(output_wav_path), samples, TARGET_SR, subtype="PCM_16")
        return float(len(samples) / TARGET_SR)

    y, sr = sf.read(str(temp_wav), dtype="float32")
    try:
        temp_wav.unlink(missing_ok=True)
    except OSError:
        pass

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Remove DC offset and normalize peak to 0.95 (FR-1.3 normalized mono PCM stream)
    if len(y) > 0:
        y = y - np.mean(y)
        peak = float(np.max(np.abs(y)))
        if peak > 1e-6:
            y = (y / peak) * 0.95

    sf.write(str(output_wav_path), y, sr, subtype="PCM_16")
    return round(float(len(y) / sr), 4)


def _get_video_duration_fallback(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 5.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 150.0
    cap.release()
    return float(frame_count / max(fps, 1.0))


def _l2_normalize(vec: np.ndarray) -> List[float]:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return [round(float(x), 6) for x in vec]


def _has_nonzero_vec(vec: Optional[List[float]], dim: int) -> bool:
    return bool(vec and len(vec) == dim and any(abs(x) > 1e-6 for x in vec))


def _blend_vectors(
    old_vec: List[float], new_vec: List[float], dim: int, alpha: float = 0.4
) -> List[float]:
    has_old = _has_nonzero_vec(old_vec, dim)
    has_new = _has_nonzero_vec(new_vec, dim)
    if has_old and has_new:
        return _l2_normalize(
            (1.0 - alpha) * np.asarray(old_vec, dtype=np.float64)
            + alpha * np.asarray(new_vec, dtype=np.float64)
        )
    if has_new:
        return list(new_vec)
    if has_old:
        return list(old_vec)
    return [0.0] * dim


def extract_beep_profile(
    y: np.ndarray, sr: int, beep_timestamp: Optional[float]
) -> BeepProfile:
    """
    Extract BeepProfile (dominant frequency, duration, spectral feature vector) around beep_timestamp.
    """
    if beep_timestamp is None or len(y) == 0:
        return BeepProfile(
            frequency_hz=3000.0,
            duration_ms=250,
            features=[0.0] * 16,
        )

    start_idx = max(0, int((beep_timestamp - 0.02) * sr))
    end_idx = min(len(y), int((beep_timestamp + 0.35) * sr))
    y_win = y[start_idx:end_idx]
    if len(y_win) < 512:
        y_win = np.pad(y_win, (0, 512 - len(y_win)))

    n_fft = 1024
    hop_length = 128
    S = np.abs(librosa.stft(y_win, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_mask = (freqs >= 800.0) & (freqs <= 5000.0)
    if np.any(band_mask):
        S_band = S[band_mask, :]
        band_freqs = freqs[band_mask]
        mean_mag = np.mean(S_band, axis=1)
        dom_freq = float(band_freqs[int(np.argmax(mean_mag))])
    else:
        dom_freq = 3000.0

    rms = librosa.feature.rms(y=y_win, frame_length=512, hop_length=hop_length)[0]
    if len(rms) > 0 and float(np.max(rms)) > 1e-6:
        active_frames = int(np.sum(rms >= 0.25 * float(np.max(rms))))
        dur_ms = int(round((active_frames * hop_length / sr) * 1000.0))
        dur_ms = max(60, min(600, dur_ms))
    else:
        dur_ms = 250

    features = extract_cf032ba_window_features(y_win, sr)
    return BeepProfile(
        frequency_hz=round(dom_freq, 2),
        duration_ms=dur_ms,
        features=features,
    )


def _compute_shot_decay_rate(y_win: np.ndarray, sr: int) -> float:
    hop_length = 64
    rms = librosa.feature.rms(y=y_win, frame_length=256, hop_length=hop_length)[0]
    if len(rms) < 3 or float(np.max(rms)) < 1e-6:
        return 120.0

    peak_frame = int(np.argmax(rms))
    tail_rms = rms[peak_frame:]
    if len(tail_rms) < 3:
        return 120.0

    db_tail = 20.0 * np.log10(np.maximum(tail_rms, 1e-6))
    times = np.arange(len(db_tail)) * (hop_length / float(sr))
    if times[-1] <= 1e-4:
        return 120.0

    slope, _ = np.polyfit(times, db_tail, 1)
    decay_rate = float(max(10.0, min(500.0, abs(slope))))
    return round(decay_rate, 3)


def extract_shot_profile(
    y: np.ndarray,
    sr: int,
    shot_timestamps: List[float],
    cap: Optional[cv2.VideoCapture] = None,
) -> ShotProfile:
    """
    Extract ShotProfile combining:
      - Original cf032ba 16-D acoustic features
      - Separate Slow (>2000ms apart) and Fast (<=2000ms apart) 16-D templates
      - Synchronized 6-D OpenCV visual recoil/flash feature vector
    """
    if not shot_timestamps or len(y) == 0:
        return ShotProfile(
            decay_rate=120.0,
            features=[0.0] * 16,
            slow_features=[0.0] * 16,
            fast_features=[0.0] * 16,
            video_features=[0.0] * 6,
        )

    sorted_shots = sorted(float(t) for t in shot_timestamps)
    decay_rates: List[float] = []
    all_feat_vecs: List[np.ndarray] = []
    slow_feat_vecs: List[np.ndarray] = []
    fast_feat_vecs: List[np.ndarray] = []
    video_vecs: List[np.ndarray] = []

    for idx, t_shot in enumerate(sorted_shots):
        delta_t = (t_shot - sorted_shots[idx - 1]) if idx > 0 else float("inf")
        is_slow = delta_t > SLOW_SHOT_THRESHOLD_SEC

        start_idx = max(0, int((t_shot - 0.015) * sr))
        end_idx = min(len(y), int((t_shot + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) < 256:
            continue

        decay_rates.append(_compute_shot_decay_rate(y_win, sr))
        feat_16 = np.asarray(
            extract_cf032ba_window_features(y_win, sr), dtype=np.float64
        )
        all_feat_vecs.append(feat_16)
        if is_slow:
            slow_feat_vecs.append(feat_16)
        else:
            fast_feat_vecs.append(feat_16)

        if cap is not None and cap.isOpened():
            v_vec = np.asarray(
                extract_video_window_features(cap, t_shot), dtype=np.float64
            )
            if np.linalg.norm(v_vec) > 1e-6:
                video_vecs.append(v_vec)

    if not all_feat_vecs:
        return ShotProfile(
            decay_rate=120.0,
            features=[0.0] * 16,
            slow_features=[0.0] * 16,
            fast_features=[0.0] * 16,
            video_features=[0.0] * 6,
        )

    mean_decay = round(float(np.mean(decay_rates)), 3)
    combined_feat = _l2_normalize(np.mean(np.stack(all_feat_vecs, axis=0), axis=0))
    slow_feat = (
        _l2_normalize(np.mean(np.stack(slow_feat_vecs, axis=0), axis=0))
        if slow_feat_vecs
        else list(combined_feat)
    )
    fast_feat = (
        _l2_normalize(np.mean(np.stack(fast_feat_vecs, axis=0), axis=0))
        if fast_feat_vecs
        else list(combined_feat)
    )
    mean_video = (
        _l2_normalize(np.mean(np.stack(video_vecs, axis=0), axis=0))
        if video_vecs
        else [0.0] * 6
    )

    return ShotProfile(
        decay_rate=mean_decay,
        features=combined_feat,
        slow_features=slow_feat,
        fast_features=fast_feat,
        video_features=mean_video,
    )


def update_signature_profile(
    existing: Optional[FirearmAudioSignature],
    gun_name: str,
    wav_path: Path,
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
    video_path: Optional[Path] = None,
    video_id: str = "session",
    base_dir: Optional[Path] = None,
) -> FirearmAudioSignature:
    """
    Create or refine a FirearmAudioSignature and harvest positive + hard-negative
    multimodal (Audio + Video) training samples for the firearm's local classifier (FR-2.4).
    """
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    new_beep = extract_beep_profile(y, sr, beep_timestamp)

    cap: Optional[cv2.VideoCapture] = None
    if video_path is not None and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))

    try:
        new_shot = extract_shot_profile(y, sr, shot_timestamps, cap=cap)
        storage_root = base_dir or Path(
            os.environ.get("ANALYZER_DATA_DIR", wav_path.parent.parent)
        )
        if shot_timestamps:
            v_count, pos_count, neg_count, cv_f1 = harvest_and_save_training_samples(
                base_dir=storage_root,
                gun_name=gun_name.strip(),
                video_id=video_id,
                y=y,
                sr=sr,
                beep_timestamp=beep_timestamp,
                confirmed_shots=shot_timestamps,
                cap=cap,
            )
            new_shot.trained_videos_count = v_count
            new_shot.positive_shots_count = pos_count
            new_shot.hard_negatives_count = neg_count
            new_shot.cv_f1_score = cv_f1
    finally:
        if cap is not None:
            cap.release()

    now_iso = datetime.now(timezone.utc).isoformat()

    if existing is None or existing.signature_version <= 0:
        return FirearmAudioSignature(
            gun_name=gun_name.strip(),
            signature_version=1,
            updated_at=now_iso,
            beep_profile=new_beep,
            shot_profile=new_shot,
        )

    alpha = 0.4

    # Blend BeepProfile
    if beep_timestamp is not None and _has_nonzero_vec(new_beep.features, 16):
        if _has_nonzero_vec(existing.beep_profile.features, 16):
            blended_freq = round(
                (1 - alpha) * existing.beep_profile.frequency_hz
                + alpha * new_beep.frequency_hz,
                2,
            )
            blended_dur = int(
                round(
                    (1 - alpha) * existing.beep_profile.duration_ms
                    + alpha * new_beep.duration_ms
                )
            )
            blended_beep_feat = _blend_vectors(
                existing.beep_profile.features, new_beep.features, 16, alpha
            )
            refined_beep = BeepProfile(
                frequency_hz=blended_freq,
                duration_ms=blended_dur,
                features=blended_beep_feat,
            )
        else:
            refined_beep = new_beep
    else:
        refined_beep = existing.beep_profile

    # Blend ShotProfile
    if shot_timestamps and _has_nonzero_vec(new_shot.features, 16):
        ex_shot = existing.shot_profile
        if _has_nonzero_vec(ex_shot.features, 16):
            blended_decay = round(
                (1 - alpha) * ex_shot.decay_rate + alpha * new_shot.decay_rate, 3
            )
            blended_feat = _blend_vectors(
                ex_shot.features, new_shot.features, 16, alpha
            )
            blended_slow = _blend_vectors(
                ex_shot.slow_features or ex_shot.features,
                new_shot.slow_features,
                16,
                alpha,
            )
            blended_fast = _blend_vectors(
                ex_shot.fast_features or ex_shot.features,
                new_shot.fast_features,
                16,
                alpha,
            )
            blended_video = _blend_vectors(
                ex_shot.video_features, new_shot.video_features, 6, alpha
            )
            refined_shot = ShotProfile(
                decay_rate=blended_decay,
                features=blended_feat,
                slow_features=blended_slow,
                fast_features=blended_fast,
                video_features=blended_video,
                trained_videos_count=new_shot.trained_videos_count,
                positive_shots_count=new_shot.positive_shots_count,
                hard_negatives_count=new_shot.hard_negatives_count,
                cv_f1_score=new_shot.cv_f1_score,
            )
        else:
            refined_shot = new_shot
    else:
        refined_shot = existing.shot_profile

    return FirearmAudioSignature(
        gun_name=existing.gun_name,
        signature_version=existing.signature_version + 1,
        updated_at=now_iso,
        beep_profile=refined_beep,
        shot_profile=refined_shot,
    )


def detect_events_with_profile(
    wav_path: Path,
    signature: FirearmAudioSignature,
    video_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[float], List[float]]:
    """
    Automated Timer Beep & Gunshot detection using the Multimodal (Video + Audio)
    and Slow (>2000ms) / Fast (<=2000ms) Supervised Detection Engine (FR-2.3).
    """
    if video_path is None:
        # Automatically discover matching video file in same directory if present
        for ext in (".mp4", ".mov", ".webm"):
            cand_v = wav_path.with_suffix(ext)
            if cand_v.exists():
                video_path = cand_v
                break

    return detect_multimodal_events(
        wav_path=wav_path,
        signature=signature,
        video_path=video_path,
        base_dir=base_dir,
    )


def detect_events_with_limit(
    wav_path: Path,
    signature: FirearmAudioSignature,
    video_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[float], List[float], Optional[float]]:
    """
    Automated detection returning (beep_timestamp, shot_timestamps, end_beep_timestamp).
    If two timer beeps are detected, the first is the start beep and the second is the time limit.
    """
    if video_path is None:
        for ext in (".mp4", ".mov", ".webm"):
            cand_v = wav_path.with_suffix(ext)
            if cand_v.exists():
                video_path = cand_v
                break

    return detect_multimodal_events_with_limit(
        wav_path=wav_path,
        signature=signature,
        video_path=video_path,
        base_dir=base_dir,
    )
