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
DUAL_REGIME_SPLIT_THRESHOLD = 0.85  # Seconds separating Isolated (>0.85s) vs Rapid Follow-up (<=0.85s)


def extract_normalized_mono_pcm(video_path: Path, output_wav_path: Path) -> float:
    """
    Extract audio track from video into a normalized mono 16-bit PCM WAV file (FR-1.3).
    Aligns audio start PTS to 0.000s (aresample=async=1:first_pts=0) to prevent A/V drift.
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


def _has_nonzero_features(vec: Optional[List[float]]) -> bool:
    return bool(vec and len(vec) == 16 and any(abs(x) > 1e-6 for x in vec))


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
    Extract a 16-dimensional acoustic feature vector for tonal signals (Timer Beep):
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


def _compute_reverb_subtracted_hf_flux(
    y: np.ndarray, sr: int, n_fft: int = 512, hop_length: int = 128
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Reverb-Subtracted High-Frequency Positive Spectral Flux (1.8 kHz - 10.5 kHz)
    and Positive Log-RMS Attack Derivative to unmask rapid follow-up shots inside heavy reverb.
    Returns:
      - hf_flux_norm: normalized reverb-subtracted HF positive flux envelope
      - attack_norm: normalized positive log-RMS derivative envelope
      - rms: raw frame RMS envelope
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = S.shape[1]

    # Causal leaky-integrator reverberation floor estimator per frequency bin
    # Tracks decaying room reverberation over ~65 ms and subtracts it from incoming frames
    reverb_floor = np.zeros_like(S)
    decay_alpha = 0.82
    for t_idx in range(1, n_frames):
        reverb_floor[:, t_idx] = (
            decay_alpha * reverb_floor[:, t_idx - 1]
            + (1.0 - decay_alpha) * S[:, t_idx - 1]
        )

    S_clean = np.maximum(0.0, S - 0.88 * reverb_floor)
    S_log = np.log1p(25.0 * S_clean)

    # Positive spectral flux on reverb-subtracted log spectrogram
    diff_S = np.diff(S_log, axis=1, prepend=S_log[:, :1])
    pos_flux = np.maximum(0.0, diff_S)

    hf_mask = (freqs >= 1800.0) & (freqs <= (sr / 2.0))
    if not np.any(hf_mask):
        hf_mask = freqs >= 1000.0

    hf_flux = np.sum(pos_flux[hf_mask, :], axis=0)
    if np.max(hf_flux) > 1e-8:
        hf_flux_norm = hf_flux / np.max(hf_flux)
    else:
        hf_flux_norm = hf_flux

    # Positive log-RMS attack derivative (captures sharp energy re-excitations during loud reverb tails)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    log_rms = np.log1p(40.0 * rms)
    attack = np.maximum(0.0, np.diff(log_rms, prepend=log_rms[:1]))
    if np.max(attack) > 1e-8:
        attack_norm = attack / np.max(attack)
    else:
        attack_norm = attack

    return hf_flux_norm, attack_norm, rms


def _extract_reverb_subtracted_shot_features(
    y: np.ndarray, sr: int, shot_t: float
) -> Tuple[List[float], float]:
    """
    Extract a 16-dimensional Reverb-Subtracted acoustic feature vector and HF flux ratio
    for a gunshot at timestamp `shot_t`:
    - Subtracts the pre-shot reverberation spectrum [shot_t - 0.085s, shot_t - 0.015s]
      from the transient shot window [shot_t - 0.012s, shot_t + 0.115s].
    - Computes 12 reverb-subtracted MFCCs + spectral centroid + bandwidth + flatness + HF flux ratio.
    """
    start_idx = max(0, int((shot_t - 0.012) * sr))
    end_idx = min(len(y), int((shot_t + 0.115) * sr))
    y_win = y[start_idx:end_idx]
    if len(y_win) < 512:
        y_win = np.pad(y_win, (0, 512 - len(y_win)))

    # Pre-shot reverberation window immediately preceding the shot onset
    pre_start = max(0, int((shot_t - 0.085) * sr))
    pre_end = max(pre_start, int((shot_t - 0.015) * sr))
    y_pre = y[pre_start:pre_end]

    n_fft = 512
    hop_length = 64
    S_shot = np.abs(librosa.stft(y_win, n_fft=n_fft, hop_length=hop_length))

    if len(y_pre) >= 256:
        S_pre = np.abs(librosa.stft(y_pre, n_fft=n_fft, hop_length=hop_length))
        reverb_profile = np.mean(S_pre, axis=1, keepdims=True)
        S_clean = np.maximum(1e-7, S_shot - 0.85 * reverb_profile)
    else:
        S_clean = np.maximum(1e-7, S_shot)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Compute High-Frequency Positive Flux Ratio (1.8 kHz - Nyquist vs Total Flux)
    S_log = np.log1p(20.0 * S_clean)
    pos_flux = np.maximum(0.0, np.diff(S_log, axis=1, prepend=S_log[:, :1]))
    hf_mask = freqs >= 1800.0
    hf_flux_sum = float(np.sum(pos_flux[hf_mask, :])) if np.any(hf_mask) else 0.0
    total_flux_sum = float(np.sum(pos_flux)) + 1e-7
    hf_flux_ratio = float(np.clip(hf_flux_sum / total_flux_sum, 0.0, 1.0))

    # Compute 12 MFCCs from reverb-subtracted power spectrogram
    power_clean = S_clean**2
    mel_spec = librosa.feature.melspectrogram(S=power_clean, sr=sr, n_mels=32)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec + 1e-10), n_mfcc=12)
    mfcc_mean = np.mean(mfcc, axis=1)

    centroid = librosa.feature.spectral_centroid(S=S_clean, sr=sr, n_fft=n_fft)
    bandwidth = librosa.feature.spectral_bandwidth(S=S_clean, sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(S=S_clean)

    raw_vec = np.concatenate(
        [
            mfcc_mean / 100.0,
            [float(np.mean(centroid)) / (sr / 2.0)],
            [float(np.mean(bandwidth)) / (sr / 2.0)],
            [float(np.mean(flatness))],
            [hf_flux_ratio],
        ]
    )
    return _l2_normalize(raw_vec), round(hf_flux_ratio, 4)


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
        band_S = S[band_mask, :]
        band_freqs = freqs[band_mask]
        mean_spectrum = np.mean(band_S, axis=1)
        dom_idx = int(np.argmax(mean_spectrum))
        dom_freq = float(band_freqs[dom_idx])

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
    Extract Dual-Regime ShotProfile (Isolated vs Rapid Follow-up reverb-subtracted features,
    decay_rate, and hf_flux_ratio) across confirmed shots.
    """
    if not shot_timestamps or len(y) == 0:
        return ShotProfile(
            decay_rate=150.0,
            features=[0.0] * 16,
            isolated_features=[0.0] * 16,
            followup_features=[0.0] * 16,
            hf_flux_ratio=0.5,
        )

    sorted_shots = sorted(float(t) for t in shot_timestamps)
    decay_rates: List[float] = []
    all_vectors: List[List[float]] = []
    isolated_vectors: List[List[float]] = []
    followup_vectors: List[List[float]] = []
    hf_ratios: List[float] = []

    for idx, shot_t in enumerate(sorted_shots):
        start_idx = max(0, int((shot_t - 0.015) * sr))
        end_idx = min(len(y), int((shot_t + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) < 256:
            continue

        decay_rates.append(_compute_shot_decay_rate(y_win, sr))
        feat, hf_ratio = _extract_reverb_subtracted_shot_features(y, sr, shot_t)
        all_vectors.append(feat)
        hf_ratios.append(hf_ratio)

        # Classify into Isolated Regime (>0.85s or first shot) vs Rapid Follow-up Regime (<=0.85s)
        if idx == 0 or (shot_t - sorted_shots[idx - 1]) > DUAL_REGIME_SPLIT_THRESHOLD:
            isolated_vectors.append(feat)
        else:
            followup_vectors.append(feat)

    if not all_vectors:
        return ShotProfile(
            decay_rate=150.0,
            features=[0.0] * 16,
            isolated_features=[0.0] * 16,
            followup_features=[0.0] * 16,
            hf_flux_ratio=0.5,
        )

    avg_decay = round(float(np.mean(decay_rates)), 3)
    avg_all = _l2_normalize(np.mean(np.asarray(all_vectors), axis=0))
    avg_iso = (
        _l2_normalize(np.mean(np.asarray(isolated_vectors), axis=0))
        if isolated_vectors
        else list(avg_all)
    )
    avg_fol = (
        _l2_normalize(np.mean(np.asarray(followup_vectors), axis=0))
        if followup_vectors
        else list(avg_all)
    )
    avg_hf = round(float(np.mean(hf_ratios)), 4) if hf_ratios else 0.5

    return ShotProfile(
        decay_rate=avg_decay,
        features=avg_all,
        isolated_features=avg_iso,
        followup_features=avg_fol,
        hf_flux_ratio=avg_hf,
    )


def _blend_feature_vectors(
    existing_vec: List[float], new_vec: List[float], alpha: float
) -> List[float]:
    if not _has_nonzero_features(new_vec):
        return existing_vec if len(existing_vec) == 16 else [0.0] * 16
    if not _has_nonzero_features(existing_vec):
        return new_vec
    return _l2_normalize(
        (1.0 - alpha) * np.asarray(existing_vec) + alpha * np.asarray(new_vec)
    )


def update_signature_profile(
    existing: Optional[FirearmAudioSignature],
    gun_name: str,
    wav_path: Path,
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
) -> FirearmAudioSignature:
    """
    Create or refine a Dual-Regime FirearmAudioSignature by averaging spectral characteristics (FR-2.4).
    """
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    new_beep = extract_beep_profile(y, sr, beep_timestamp)
    new_shot = extract_shot_profile(y, sr, shot_timestamps)
    now_iso = datetime.now(timezone.utc).isoformat()

    if existing is None or existing.signature_version <= 0:
        return FirearmAudioSignature(
            gun_name=gun_name.strip(),
            signature_version=1,
            updated_at=now_iso,
            beep_profile=new_beep,
            shot_profile=new_shot,
        )

    alpha = 0.4  # Learning rate for refining profile with latest confirmed session

    # Blend BeepProfile if beep_timestamp was provided
    if beep_timestamp is not None and _has_nonzero_features(new_beep.features):
        if _has_nonzero_features(existing.beep_profile.features):
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
            blended_beep_feat = _blend_feature_vectors(
                existing.beep_profile.features, new_beep.features, alpha
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

    # Blend Dual-Regime ShotProfile if shot_timestamps were provided
    if shot_timestamps and _has_nonzero_features(new_shot.features):
        if _has_nonzero_features(existing.shot_profile.features):
            blended_decay = round(
                (1 - alpha) * existing.shot_profile.decay_rate
                + alpha * new_shot.decay_rate,
                3,
            )
            blended_hf = round(
                (1 - alpha) * getattr(existing.shot_profile, "hf_flux_ratio", 0.5)
                + alpha * new_shot.hf_flux_ratio,
                4,
            )
            blended_all = _blend_feature_vectors(
                existing.shot_profile.features, new_shot.features, alpha
            )
            blended_iso = _blend_feature_vectors(
                existing.shot_profile.isolated_features or existing.shot_profile.features,
                new_shot.isolated_features,
                alpha,
            )
            blended_fol = _blend_feature_vectors(
                existing.shot_profile.followup_features or existing.shot_profile.features,
                new_shot.followup_features,
                alpha,
            )
            refined_shot = ShotProfile(
                decay_rate=blended_decay,
                features=blended_all,
                isolated_features=blended_iso,
                followup_features=blended_fol,
                hf_flux_ratio=blended_hf,
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
    Automated Dual-Regime peak detection and Reverb-Subtracted High-Frequency Flux
    template matching using librosa (FR-2.3).
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

    # 1. Detect Timer Beep using narrowband frequency match + tonal purity
    target_freq = signature.beep_profile.frequency_hz or 3000.0
    beep_band = np.abs(freqs - target_freq) <= 350.0
    if not np.any(beep_band):
        beep_band = (freqs >= 800.0) & (freqs <= 5000.0)

    band_energy = np.sum(S[beep_band, :], axis=0)
    total_energy = np.sum(S, axis=0) + 1e-7
    band_ratio = band_energy / total_energy

    tonal_score = band_energy * band_ratio * (1.0 - np.clip(flatness, 0.0, 1.0))

    dur_frames = max(
        3, int(round((signature.beep_profile.duration_ms / 1000.0) * sr / hop_length))
    )
    kernel = np.ones(dur_frames, dtype=np.float32) / float(dur_frames)
    smoothed_tonal = np.convolve(tonal_score, kernel, mode="same")

    beep_timestamp: Optional[float] = None
    if np.max(smoothed_tonal) > 1e-4:
        peak_idx = int(np.argmax(smoothed_tonal))
        threshold = 0.30 * smoothed_tonal[peak_idx]
        onset_frame = peak_idx
        while onset_frame > 0 and smoothed_tonal[onset_frame - 1] >= threshold:
            onset_frame -= 1
        beep_timestamp = round(
            float(librosa.frames_to_time(onset_frame, sr=sr, hop_length=hop_length)), 3
        )

    # 2. Detect Gunshots using Reverb-Subtracted High-Frequency Flux + Dual-Regime Matching
    hf_flux_norm, attack_norm, rms = _compute_reverb_subtracted_hf_flux(
        y, sr, n_fft=512, hop_length=hop_length
    )
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if np.max(onset_env) > 1e-8:
        onset_norm = onset_env / np.max(onset_env)
    else:
        onset_norm = onset_env

    n_frames = min(len(hf_flux_norm), len(attack_norm), len(rms), len(onset_norm))
    hf_flux_norm = hf_flux_norm[:n_frames]
    attack_norm = attack_norm[:n_frames]
    rms = rms[:n_frames]
    onset_norm = onset_norm[:n_frames]

    rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0
    rms_norm = rms / max(rms_max, 1e-8)

    # Composite transient score: combines Reverb-Subtracted HF Flux, Log-RMS Attack, and RMS/Onset
    # Ensures both isolated shots AND rapid follow-up shots riding on reverb tails produce clear peaks
    composite_env = rms_norm * (
        0.50 * hf_flux_norm + 0.30 * attack_norm + 0.20 * onset_norm + 0.25
    )
    if np.max(composite_env) > 1e-8:
        composite_env = composite_env / np.max(composite_env)

    min_wait_frames = max(1, int(0.11 * sr / hop_length))
    peaks = librosa.util.peak_pick(
        composite_env,
        pre_max=3,
        post_max=3,
        pre_avg=6,
        post_avg=6,
        delta=0.09,
        wait=min_wait_frames,
    )

    shot_timestamps: List[float] = []
    min_shot_time = (beep_timestamp + 0.18) if beep_timestamp is not None else 0.05

    iso_ref = (
        signature.shot_profile.isolated_features
        if _has_nonzero_features(signature.shot_profile.isolated_features)
        else signature.shot_profile.features
    )
    fol_ref = (
        signature.shot_profile.followup_features
        if _has_nonzero_features(signature.shot_profile.followup_features)
        else iso_ref
    )
    all_ref = signature.shot_profile.features
    has_ref = _has_nonzero_features(all_ref) or _has_nonzero_features(iso_ref)

    for p_frame in peaks:
        p_idx = int(p_frame)
        cand_time = float(librosa.frames_to_time(p_idx, sr=sr, hop_length=hop_length))
        if cand_time < min_shot_time:
            continue

        if rms[p_idx] < 0.14 * rms_max:
            continue

        # Determine active regime based on time since previous detected shot
        is_followup = (
            len(shot_timestamps) > 0
            and (cand_time - shot_timestamps[-1]) <= DUAL_REGIME_SPLIT_THRESHOLD
        )

        if is_followup:
            # In Rapid Follow-up Regime, verify there is a genuine HF Flux / Attack re-excitation
            # (rejects passive room echo peaks that lack a high-frequency muzzle crack)
            win_lo = max(0, p_idx - 2)
            win_hi = min(n_frames, p_idx + 3)
            local_hf_flux = float(np.max(hf_flux_norm[win_lo:win_hi]))
            local_attack = float(np.max(attack_norm[win_lo:win_hi]))
            if local_hf_flux < 0.08 and local_attack < 0.08:
                continue

        if has_ref:
            cand_feat, _ = _extract_reverb_subtracted_shot_features(y, sr, cand_time)
            active_ref = fol_ref if is_followup else iso_ref
            sim_regime = _cosine_similarity(cand_feat, active_ref)
            sim_global = _cosine_similarity(cand_feat, all_ref)
            best_sim = max(sim_regime, sim_global)
            min_sim_threshold = 0.28 if is_followup else 0.33
            if best_sim < min_sim_threshold:
                continue

        # Refine to exact local transient peak around candidate onset
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
