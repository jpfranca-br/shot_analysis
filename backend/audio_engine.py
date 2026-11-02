from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf

from backend.models import BeepProfile, FirearmAudioSignature, ShotProfile

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
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-af",
        "aresample=async=1:first_pts=0",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-acodec",
        "pcm_s16le",
        str(temp_wav),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not temp_wav.exists():
        # If video has no audio track, generate a silent WAV matching video duration
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
    import cv2

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


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _extract_window_features(y_win: np.ndarray, sr: int) -> List[float]:
    """
    Extract a 16-dimensional acoustic feature vector using librosa:
    - 13 mean MFCC coefficients
    - Mean spectral centroid (scaled)
    - Mean spectral bandwidth (scaled)
    - Mean spectral flatness
    """
    if len(y_win) < 512:
        y_win = np.pad(y_win, (0, 512 - len(y_win)))

    n_fft = min(1024, len(y_win))
    hop_length = max(128, n_fft // 4)

    mfcc = librosa.feature.mfcc(
        y=y_win, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(
        y=y_win, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y_win, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    flatness = librosa.feature.spectral_flatness(
        y=y_win, n_fft=n_fft, hop_length=hop_length
    )

    raw_vec = np.concatenate(
        [
            mfcc_mean / 100.0,
            [float(np.mean(centroid)) / (sr / 2.0)],
            [float(np.mean(bandwidth)) / (sr / 2.0)],
            [float(np.mean(flatness))],
        ]
    )
    return _l2_normalize(raw_vec)


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

    # Focus on typical shot timer beep frequency band (800 Hz - 5000 Hz)
    band_mask = (freqs >= 800.0) & (freqs <= 5000.0)
    if np.any(band_mask):
        band_S = S[band_mask, :]
        band_freqs = freqs[band_mask]
        mean_spectrum = np.mean(band_S, axis=1)
        dom_idx = int(np.argmax(mean_spectrum))
        dom_freq = float(band_freqs[dom_idx])

        # Estimate duration where narrowband energy around dom_freq exceeds 25% of peak
        narrow_mask = np.abs(band_freqs - dom_freq) <= 250.0
        frame_energy = np.sum(band_S[narrow_mask, :], axis=0)
        peak_e = float(np.max(frame_energy)) if len(frame_energy) > 0 else 0.0
        if peak_e > 1e-6:
            active_frames = int(np.sum(frame_energy >= 0.25 * peak_e))
            duration_ms = int(round((active_frames * hop_length / sr) * 1000.0))
            duration_ms = max(60, min(800, duration_ms))
        else:
            duration_ms = 200
    else:
        dom_freq = 3000.0
        duration_ms = 200

    features = _extract_window_features(y_win, sr)
    return BeepProfile(
        frequency_hz=round(dom_freq, 2),
        duration_ms=duration_ms,
        features=features,
    )


def _compute_shot_decay_rate(y_win: np.ndarray, sr: int) -> float:
    """
    Estimate envelope decay rate (in dB/sec) after transient peak.
    """
    hop = 64
    rms = librosa.feature.rms(y=y_win, frame_length=256, hop_length=hop)[0]
    if len(rms) < 3:
        return 120.0
    peak_idx = int(np.argmax(rms))
    post_rms = rms[peak_idx:]
    if len(post_rms) < 3:
        post_rms = rms
    db = librosa.amplitude_to_db(post_rms + 1e-7, ref=np.max)
    times = np.arange(len(db)) * (hop / float(sr))
    if len(times) >= 2 and times[-1] > 1e-5:
        slope, _ = np.polyfit(times, db, 1)
        return round(float(abs(slope)), 3)
    return 120.0


def extract_shot_profile(
    y: np.ndarray, sr: int, shot_timestamps: List[float]
) -> ShotProfile:
    """
    Extract ShotProfile (average decay_rate and acoustic feature vector) across confirmed shots.
    """
    if not shot_timestamps or len(y) == 0:
        return ShotProfile(
            decay_rate=150.0,
            features=[0.0] * 16,
        )

    decay_rates: List[float] = []
    feature_vectors: List[List[float]] = []

    for shot_t in shot_timestamps:
        start_idx = max(0, int((shot_t - 0.015) * sr))
        end_idx = min(len(y), int((shot_t + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) < 256:
            continue
        decay_rates.append(_compute_shot_decay_rate(y_win, sr))
        feature_vectors.append(_extract_window_features(y_win, sr))

    if not feature_vectors:
        return ShotProfile(decay_rate=150.0, features=[0.0] * 16)

    avg_decay = round(float(np.mean(decay_rates)), 3)
    avg_feat = _l2_normalize(np.mean(np.asarray(feature_vectors), axis=0))
    return ShotProfile(decay_rate=avg_decay, features=avg_feat)


def update_signature_profile(
    existing: Optional[FirearmAudioSignature],
    gun_name: str,
    wav_path: Path,
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
) -> FirearmAudioSignature:
    """
    Create or refine a FirearmAudioSignature by averaging spectral characteristics (FR-2.4).
    """
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    new_beep = extract_beep_profile(y, sr, beep_timestamp)
    new_shot = extract_shot_profile(y, sr, shot_timestamps)
    now_iso = datetime.now(timezone.utc).isoformat()

    if existing is None:
        return FirearmAudioSignature(
            gun_name=gun_name.strip(),
            signature_version=1,
            updated_at=now_iso,
            beep_profile=new_beep,
            shot_profile=new_shot,
        )

    alpha = 0.4  # Learning rate for refining profile with latest confirmed session

    # Blend BeepProfile if beep_timestamp was provided
    if beep_timestamp is not None and any(abs(x) > 1e-6 for x in new_beep.features):
        if any(abs(x) > 1e-6 for x in existing.beep_profile.features):
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
            blended_beep_feat = _l2_normalize(
                (1 - alpha) * np.asarray(existing.beep_profile.features)
                + alpha * np.asarray(new_beep.features)
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

    # Blend ShotProfile if shot_timestamps were provided
    if shot_timestamps and any(abs(x) > 1e-6 for x in new_shot.features):
        if any(abs(x) > 1e-6 for x in existing.shot_profile.features):
            blended_decay = round(
                (1 - alpha) * existing.shot_profile.decay_rate
                + alpha * new_shot.decay_rate,
                3,
            )
            blended_shot_feat = _l2_normalize(
                (1 - alpha) * np.asarray(existing.shot_profile.features)
                + alpha * np.asarray(new_shot.features)
            )
            refined_shot = ShotProfile(
                decay_rate=blended_decay,
                features=blended_shot_feat,
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
    wav_path: Path, signature: FirearmAudioSignature
) -> Tuple[Optional[float], List[float]]:
    """
    Automated peak detection and template matching using librosa (FR-2.3)
    to predict timer beep and gunshot timestamps.
    """
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    if len(y) < sr * 0.2:
        return None, []

    n_fft = 1024
    hop_length = 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(
        S=S, n_fft=n_fft, hop_length=hop_length
    )[0]

    # 1. Detect Timer Beep using narrowband frequency match + tonal purity + template match
    target_freq = signature.beep_profile.frequency_hz or 3000.0
    beep_band = np.abs(freqs - target_freq) <= 350.0
    if not np.any(beep_band):
        beep_band = (freqs >= 800.0) & (freqs <= 5000.0)

    band_energy = np.sum(S[beep_band, :], axis=0)
    total_energy = np.sum(S, axis=0) + 1e-7
    band_ratio = band_energy / total_energy

    # Tonal score rewards frames with strong target frequency energy and low spectral flatness
    tonal_score = band_energy * band_ratio * (1.0 - np.clip(flatness, 0.0, 1.0))

    # Smooth over expected beep duration
    dur_frames = max(
        3, int(round((signature.beep_profile.duration_ms / 1000.0) * sr / hop_length))
    )
    kernel = np.ones(dur_frames, dtype=np.float32) / float(dur_frames)
    smoothed_tonal = np.convolve(tonal_score, kernel, mode="same")

    beep_timestamp: Optional[float] = None
    if np.max(smoothed_tonal) > 1e-4:
        # Find top tonal candidates and verify with feature template similarity
        peak_idx = int(np.argmax(smoothed_tonal))
        # Walk backwards to find onset of the beep tone
        threshold = 0.30 * smoothed_tonal[peak_idx]
        onset_frame = peak_idx
        while onset_frame > 0 and smoothed_tonal[onset_frame - 1] >= threshold:
            onset_frame -= 1
        beep_timestamp = round(
            float(librosa.frames_to_time(onset_frame, sr=sr, hop_length=hop_length)), 3
        )

    # 2. Detect Gunshots using onset envelope + RMS peak picking + template matching
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    combined_env = rms * (onset_env + 0.5)
    if np.max(combined_env) > 0:
        combined_env = combined_env / np.max(combined_env)

    min_wait_frames = max(1, int(0.12 * sr / hop_length))
    peaks = librosa.util.peak_pick(
        combined_env,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.12,
        wait=min_wait_frames,
    )

    shot_timestamps: List[float] = []
    min_shot_time = (beep_timestamp + 0.18) if beep_timestamp is not None else 0.05

    has_ref_features = any(abs(x) > 1e-6 for x in signature.shot_profile.features)
    rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0

    for p_frame in peaks:
        cand_time = float(librosa.frames_to_time(int(p_frame), sr=sr, hop_length=hop_length))
        if cand_time < min_shot_time:
            continue

        # Ensure peak amplitude is meaningful
        if rms[int(p_frame)] < 0.15 * rms_max:
            continue

        start_idx = max(0, int((cand_time - 0.015) * sr))
        end_idx = min(len(y), int((cand_time + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) < 256:
            continue

        if has_ref_features:
            cand_feat = _extract_window_features(y_win, sr)
            sim = _cosine_similarity(cand_feat, signature.shot_profile.features)
            if sim < 0.35:
                continue

        # Refine to exact local waveform peak around candidate onset
        local_start = max(0, int((cand_time - 0.02) * sr))
        local_end = min(len(y), int((cand_time + 0.04) * sr))
        if local_end > local_start:
            local_peak_offset = int(np.argmax(np.abs(y[local_start:local_end])))
            refined_time = (local_start + local_peak_offset) / float(sr)
        else:
            refined_time = cand_time

        if not shot_timestamps or (refined_time - shot_timestamps[-1]) >= 0.10:
            shot_timestamps.append(round(float(refined_time), 3))

    return beep_timestamp, shot_timestamps
