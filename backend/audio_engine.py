from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import List, Optional, Tuple

import imageio_ffmpeg
import librosa
import numpy as np
import soundfile as sf

from backend.models import BeepProfile, FirearmAudioSignature, ShotProfile
from backend.yamnet_model import YAMNetClassifier

TARGET_SR = 22050
SLOW_SHOT_THRESHOLD_SEC = 2.000  # 2000ms threshold separating Slow (>2000ms) vs Fast (<=2000ms) shots


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


def _has_nonzero_vec(vec: Optional[List[float]], expected_len: int) -> bool:
    return bool(
        vec and len(vec) == expected_len and any(abs(x) > 1e-6 for x in vec)
    )


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


def _extract_regime_acoustic_features(
    y: np.ndarray, sr: int, shot_t: float, regime: str
) -> List[float]:
    """
    Extract 16-D acoustic features tailored to Slow (>2000ms) vs Fast (<=2000ms) shots:
    - Slow shot (>2000ms apart or first shot): [shot_t - 0.020s, shot_t + 0.250s] (270ms window
      capturing full uncompressed attack + ballistic room decay).
    - Fast shot (<=2000ms apart): [shot_t - 0.012s, shot_t + 0.125s] (137ms tight window
      avoiding adjacent shots while capturing AGC-compressed muzzle crack).
    """
    if regime == "slow":
        start_idx = max(0, int((shot_t - 0.020) * sr))
        end_idx = min(len(y), int((shot_t + 0.250) * sr))
    else:
        start_idx = max(0, int((shot_t - 0.012) * sr))
        end_idx = min(len(y), int((shot_t + 0.125) * sr))

    y_win = y[start_idx:end_idx]
    return _extract_window_features(y_win, sr)


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
    Extract ShotProfile using Google YAMNet local neural embeddings + acoustic features,
    explicitly separating Slow Shots (>2000ms apart or first shot) and Fast Shots (<=2000ms apart).
    """
    if not shot_timestamps or len(y) == 0:
        return ShotProfile(
            decay_rate=150.0,
            features=[0.0] * 16,
            slow_features=[0.0] * 16,
            fast_features=[0.0] * 16,
            neural_embedding=[0.0] * 64,
            slow_neural_embedding=[0.0] * 64,
            fast_neural_embedding=[0.0] * 64,
            yamnet_gunshot_score=0.0,
        )

    sorted_shots = sorted(float(t) for t in shot_timestamps)
    regimes: List[str] = []
    for idx, shot_t in enumerate(sorted_shots):
        # First shot or any shot > 2000ms (2.0s) after previous shot has a slow/uncompressed signature
        if idx == 0 or (shot_t - sorted_shots[idx - 1]) > SLOW_SHOT_THRESHOLD_SEC:
            regimes.append("slow")
        else:
            regimes.append("fast")

    yamnet = YAMNetClassifier.get_instance()
    neural_results = yamnet.analyze_windows(y, sr, sorted_shots, regimes=regimes)

    decay_rates: List[float] = []
    all_feat_16: List[List[float]] = []
    slow_feat_16: List[List[float]] = []
    fast_feat_16: List[List[float]] = []

    all_emb_64: List[List[float]] = []
    slow_emb_64: List[List[float]] = []
    fast_emb_64: List[List[float]] = []
    yamnet_scores: List[float] = []

    for idx, shot_t in enumerate(sorted_shots):
        regime = regimes[idx]
        start_idx = max(0, int((shot_t - 0.015) * sr))
        end_idx = min(len(y), int((shot_t + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) >= 256:
            decay_rates.append(_compute_shot_decay_rate(y_win, sr))

        feat_16 = _extract_regime_acoustic_features(y, sr, shot_t, regime)
        all_feat_16.append(feat_16)
        if regime == "slow":
            slow_feat_16.append(feat_16)
        else:
            fast_feat_16.append(feat_16)

        if idx < len(neural_results):
            n_res = neural_results[idx]
            emb_64 = list(n_res["embedding_64d"])
            all_emb_64.append(emb_64)
            yamnet_scores.append(float(n_res["gunshot_prob"]))
            if regime == "slow":
                slow_emb_64.append(emb_64)
            else:
                fast_emb_64.append(emb_64)

    avg_decay = round(float(np.mean(decay_rates)), 3) if decay_rates else 150.0
    avg_feat_all = _l2_normalize(np.mean(np.asarray(all_feat_16), axis=0))
    avg_feat_slow = (
        _l2_normalize(np.mean(np.asarray(slow_feat_16), axis=0))
        if slow_feat_16
        else list(avg_feat_all)
    )
    avg_feat_fast = (
        _l2_normalize(np.mean(np.asarray(fast_feat_16), axis=0))
        if fast_feat_16
        else list(avg_feat_all)
    )

    avg_emb_all = (
        _l2_normalize(np.mean(np.asarray(all_emb_64), axis=0))
        if all_emb_64
        else [0.0] * 64
    )
    avg_emb_slow = (
        _l2_normalize(np.mean(np.asarray(slow_emb_64), axis=0))
        if slow_emb_64
        else list(avg_emb_all)
    )
    avg_emb_fast = (
        _l2_normalize(np.mean(np.asarray(fast_emb_64), axis=0))
        if fast_emb_64
        else list(avg_emb_all)
    )
    avg_yamnet = round(float(np.mean(yamnet_scores)), 4) if yamnet_scores else 0.5

    return ShotProfile(
        decay_rate=avg_decay,
        features=avg_feat_all,
        slow_features=avg_feat_slow,
        fast_features=avg_feat_fast,
        neural_embedding=avg_emb_all,
        slow_neural_embedding=avg_emb_slow,
        fast_neural_embedding=avg_emb_fast,
        yamnet_gunshot_score=avg_yamnet,
    )


def _blend_vectors(
    existing_vec: List[float], new_vec: List[float], expected_len: int, alpha: float
) -> List[float]:
    if not _has_nonzero_vec(new_vec, expected_len):
        return (
            existing_vec
            if _has_nonzero_vec(existing_vec, expected_len)
            else [0.0] * expected_len
        )
    if not _has_nonzero_vec(existing_vec, expected_len):
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
    Create or refine a FirearmAudioSignature with Google YAMNet neural embeddings and
    separate Slow (>2000ms) / Fast (<=2000ms) shot templates (FR-2.4).
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

    # Blend ShotProfile (Slow >2000ms and Fast <=2000ms acoustic + YAMNet neural vectors)
    if shot_timestamps and _has_nonzero_vec(new_shot.features, 16):
        ex_shot = existing.shot_profile
        if _has_nonzero_vec(ex_shot.features, 16):
            blended_decay = round(
                (1 - alpha) * ex_shot.decay_rate + alpha * new_shot.decay_rate, 3
            )
            blended_yamnet = round(
                (1 - alpha) * getattr(ex_shot, "yamnet_gunshot_score", 0.5)
                + alpha * new_shot.yamnet_gunshot_score,
                4,
            )
            blended_feat = _blend_vectors(ex_shot.features, new_shot.features, 16, alpha)
            blended_slow_feat = _blend_vectors(
                ex_shot.slow_features or ex_shot.features,
                new_shot.slow_features,
                16,
                alpha,
            )
            blended_fast_feat = _blend_vectors(
                ex_shot.fast_features or ex_shot.features,
                new_shot.fast_features,
                16,
                alpha,
            )
            blended_emb = _blend_vectors(
                ex_shot.neural_embedding, new_shot.neural_embedding, 64, alpha
            )
            blended_slow_emb = _blend_vectors(
                ex_shot.slow_neural_embedding or ex_shot.neural_embedding,
                new_shot.slow_neural_embedding,
                64,
                alpha,
            )
            blended_fast_emb = _blend_vectors(
                ex_shot.fast_neural_embedding or ex_shot.neural_embedding,
                new_shot.fast_neural_embedding,
                64,
                alpha,
            )
            refined_shot = ShotProfile(
                decay_rate=blended_decay,
                features=blended_feat,
                slow_features=blended_slow_feat,
                fast_features=blended_fast_feat,
                neural_embedding=blended_emb,
                slow_neural_embedding=blended_slow_emb,
                fast_neural_embedding=blended_fast_emb,
                yamnet_gunshot_score=blended_yamnet,
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
    Automated Timer Beep & Gunshot detection combining:
      1. Google YAMNet Local Neural Classification & 64-D Embeddings (ONNX Runtime)
      2. Slow Shots (>2000ms apart / first shot) vs. Fast Shots (<=2000ms apart) signature matching
      3. Librosa onset & spectral envelope analysis (FR-2.3)
    """
    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    if len(y) < sr * 0.2:
        return None, []

    yamnet = YAMNetClassifier.get_instance()

    n_fft = 1024
    hop_length = 128
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(
        S=S, n_fft=n_fft, hop_length=hop_length
    )[0]

    # 1. Detect Timer Beep using narrowband frequency match + tonal purity + YAMNet Beep verification
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
        # Evaluate top tonal peaks with YAMNet beep classifier + beep feature template
        tonal_norm = smoothed_tonal / np.max(smoothed_tonal)
        beep_peaks = librosa.util.peak_pick(
            tonal_norm,
            pre_max=5,
            post_max=5,
            pre_avg=10,
            post_avg=10,
            delta=0.20,
            wait=int(0.25 * sr / hop_length),
        )
        if len(beep_peaks) == 0:
            beep_peaks = np.array([int(np.argmax(smoothed_tonal))])

        cand_beep_times = []
        cand_onset_frames = []
        for bp in beep_peaks[:8]:
            bp_idx = int(bp)
            thresh = 0.30 * smoothed_tonal[bp_idx]
            onset_f = bp_idx
            while onset_f > 0 and smoothed_tonal[onset_f - 1] >= thresh:
                onset_f -= 1
            cand_onset_frames.append(onset_f)
            cand_beep_times.append(
                float(librosa.frames_to_time(onset_f, sr=sr, hop_length=hop_length))
            )

        beep_neural = yamnet.analyze_windows(
            y, sr, cand_beep_times, regimes=["slow"] * len(cand_beep_times)
        )
        has_beep_feat = _has_nonzero_vec(signature.beep_profile.features, 16)

        best_beep_score = -1.0
        best_beep_time = cand_beep_times[0]
        for idx, b_time in enumerate(cand_beep_times):
            t_score = float(tonal_norm[int(beep_peaks[idx])])
            n_beep = float(beep_neural[idx]["beep_prob"]) if idx < len(beep_neural) else 0.0
            f_sim = 0.5
            if has_beep_feat:
                s_i = max(0, int((b_time - 0.02) * sr))
                e_i = min(len(y), int((b_time + 0.35) * sr))
                f_sim = _cosine_similarity(
                    _extract_window_features(y[s_i:e_i], sr),
                    signature.beep_profile.features,
                )
            combined_b = 0.55 * t_score + 0.30 * f_sim + 0.15 * n_beep
            if combined_b > best_beep_score:
                best_beep_score = combined_b
                best_beep_time = b_time

        beep_timestamp = round(float(best_beep_time), 3)

    # 2. Propose Candidate Gunshots using RMS + Onset Envelope
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    n_frames = min(len(rms), len(onset_env))
    rms = rms[:n_frames]
    onset_env = onset_env[:n_frames]

    rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0
    combined_env = rms * (onset_env + 0.5)
    if np.max(combined_env) > 0:
        combined_env = combined_env / np.max(combined_env)

    min_wait_frames = max(1, int(0.11 * sr / hop_length))
    peaks = librosa.util.peak_pick(
        combined_env,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.08,
        wait=min_wait_frames,
    )

    min_shot_time = (beep_timestamp + 0.18) if beep_timestamp is not None else 0.05
    cand_frames: List[int] = []
    cand_times: List[float] = []

    for p_frame in peaks:
        p_idx = int(p_frame)
        cand_t = float(librosa.frames_to_time(p_idx, sr=sr, hop_length=hop_length))
        if cand_t < min_shot_time:
            continue
        if rms[p_idx] < 0.12 * rms_max:
            continue
        cand_frames.append(p_idx)
        cand_times.append(cand_t)

    if not cand_times:
        return beep_timestamp, []

    # 3. Run Batched Google YAMNet Neural Inference for Slow (>2000ms) and Fast (<=2000ms) Windows
    slow_neural_list = yamnet.analyze_windows(
        y, sr, cand_times, regimes=["slow"] * len(cand_times)
    )
    fast_neural_list = yamnet.analyze_windows(
        y, sr, cand_times, regimes=["fast"] * len(cand_times)
    )

    sp = signature.shot_profile
    slow_feat_ref = (
        sp.slow_features
        if _has_nonzero_vec(sp.slow_features, 16)
        else sp.features
    )
    fast_feat_ref = (
        sp.fast_features
        if _has_nonzero_vec(sp.fast_features, 16)
        else slow_feat_ref
    )
    has_acoustic_ref = _has_nonzero_vec(slow_feat_ref, 16)

    slow_emb_ref = (
        sp.slow_neural_embedding
        if _has_nonzero_vec(sp.slow_neural_embedding, 64)
        else sp.neural_embedding
    )
    fast_emb_ref = (
        sp.fast_neural_embedding
        if _has_nonzero_vec(sp.fast_neural_embedding, 64)
        else slow_emb_ref
    )
    has_neural_ref = _has_nonzero_vec(slow_emb_ref, 64)

    shot_timestamps: List[float] = []

    # 4. Sequential Slow (>2000ms) vs Fast (<=2000ms) Neural + Acoustic Evaluation
    for idx, cand_time in enumerate(cand_times):
        p_idx = cand_frames[idx]
        cand_rms_ratio = float(rms[p_idx] / max(rms_max, 1e-8))

        # Check time elapsed since last accepted shot (2000ms = 2.0s threshold)
        delta_since_last = (
            (cand_time - shot_timestamps[-1])
            if len(shot_timestamps) > 0
            else float("inf")
        )
        is_slow_regime = delta_since_last > SLOW_SHOT_THRESHOLD_SEC
        regime_name = "slow" if is_slow_regime else "fast"

        n_info = slow_neural_list[idx] if is_slow_regime else fast_neural_list[idx]
        gunshot_prob = max(
            float(slow_neural_list[idx]["gunshot_prob"]),
            float(fast_neural_list[idx]["gunshot_prob"]),
        )
        speech_prob = float(n_info["speech_prob"])
        cand_emb_64 = list(n_info["embedding_64d"])

        # Reject voice/speech chatter unless there is strong gunshot impulse probability
        if speech_prob > 0.45 and gunshot_prob < 0.20:
            continue

        # Regime-specific amplitude threshold:
        # Slow shots (>2000ms apart) have full uncompressed mic gain;
        # Fast shots (<=2000ms apart) experience mic AGC compression.
        min_rms_req = 0.16 if is_slow_regime else 0.12
        if cand_rms_ratio < min_rms_req:
            continue

        # Acoustic similarity (16-D MFCC + spectral vector)
        acoustic_sim = 0.5
        if has_acoustic_ref:
            cand_feat_16 = _extract_regime_acoustic_features(
                y, sr, cand_time, regime_name
            )
            active_feat_ref = slow_feat_ref if is_slow_regime else fast_feat_ref
            sim_reg = _cosine_similarity(cand_feat_16, active_feat_ref)
            sim_all = _cosine_similarity(cand_feat_16, sp.features)
            acoustic_sim = max(sim_reg, sim_all)

        # Neural embedding similarity (64-D Google YAMNet embedding)
        neural_sim = 0.5
        if has_neural_ref:
            active_emb_ref = slow_emb_ref if is_slow_regime else fast_emb_ref
            sim_emb_reg = _cosine_similarity(cand_emb_64, active_emb_ref)
            sim_emb_all = (
                _cosine_similarity(cand_emb_64, sp.neural_embedding)
                if _has_nonzero_vec(sp.neural_embedding, 64)
                else sim_emb_reg
            )
            neural_sim = max(sim_emb_reg, sim_emb_all)

        if has_neural_ref and has_acoustic_ref:
            combined_score = (
                0.45 * neural_sim
                + 0.35 * acoustic_sim
                + 0.20 * max(gunshot_prob, cand_rms_ratio)
            )
            min_score = 0.42 if is_slow_regime else 0.38
            if combined_score < min_score or acoustic_sim < 0.32:
                continue
        elif has_acoustic_ref:
            # Backward-compatible fallback for profiles trained before neural embedding storage
            combined_score = 0.65 * acoustic_sim + 0.35 * max(gunshot_prob, cand_rms_ratio)
            min_score = 0.40 if is_slow_regime else 0.36
            if combined_score < min_score or acoustic_sim < 0.30:
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
