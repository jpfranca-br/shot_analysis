"""
Multimodal (Video + Audio) Beep & Gunshot Detection and Per-Firearm Supervised Learning Engine.

Combines:
  1. Original cf032ba 16-D acoustic feature representation (13 MFCCs + Centroid + Bandwidth + Flatness)
  2. Narrowband Spectral-Crest + Video Draw-Motion Timer Beep Detection
  3. Slow Shots (>2000ms apart) vs. Fast Follow-up Shots (<=2000ms apart) Reverb Step-Up Physics
  4. OpenCV Frame-Synchronized Visual Recoil Jerk, Motion Blur Shock & Muzzle Flash Extraction
  5. Per-Firearm Positive + Hard-Negative Dataset Harvesting & Local Random Forest Classifier
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from backend.models import FirearmAudioSignature, SessionCheckResult

SLOW_SHOT_THRESHOLD_SEC = 2.000  # 2000ms boundary between slow and fast shots


def _l2_normalize(vec: np.ndarray) -> List[float]:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return [round(float(x), 6) for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def extract_cf032ba_window_features(y_win: np.ndarray, sr: int) -> List[float]:
    """
    Exact original cf032ba 16-dimensional acoustic feature vector:
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


def extract_video_window_features(
    cap: Optional[cv2.VideoCapture],
    center_time: float,
    pre_sec: float = 0.09,
    post_sec: float = 0.09,
) -> List[float]:
    """
    Extract a 6-D synchronized visual feature vector around `center_time` using OpenCV PTS timestamps:
      [0] diff_mean_norm: Mean frame-to-frame motion in window
      [1] diff_p99_norm:  99th-percentile localized pixel motion (hands / slide / muzzle)
      [2] recoil_jerk:    Max frame-to-frame acceleration of diff_p99 (recoil snap)
      [3] blur_shock:     Max relative drop in Laplacian variance (sensor shockwave blur)
      [4] flash_delta:    Max positive 99.5th-percentile luminance spike (muzzle flash / gas)
      [5] motion_ratio:   Ratio of peak recoil motion to window baseline motion
    """
    if cap is None or not cap.isOpened():
        return [0.0] * 6

    start_ms = max(0.0, (center_time - pre_sec) * 1000.0)
    end_ms = (center_time + post_sec) * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)

    prev_gray: Optional[np.ndarray] = None
    diff_means: List[float] = []
    diff_p99s: List[float] = []
    lap_vars: List[float] = []
    flash_spikes: List[float] = []

    max_frames = 16
    frames_read = 0
    while frames_read < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames_read += 1
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap_vars.append(float(np.var(lap)))

        if prev_gray is not None:
            delta = gray - prev_gray
            abs_d = np.abs(delta)
            diff_means.append(float(np.mean(abs_d)))
            diff_p99s.append(float(np.percentile(abs_d, 99)))
            pos_flash = float(np.percentile(np.maximum(0.0, delta), 99.5))
            flash_spikes.append(pos_flash)

        prev_gray = gray
        if t_ms > end_ms and len(diff_p99s) >= 2:
            break

    if not diff_p99s:
        return [0.0] * 6

    arr_mean = np.asarray(diff_means, dtype=np.float32)
    arr_p99 = np.asarray(diff_p99s, dtype=np.float32)
    arr_lap = np.asarray(lap_vars, dtype=np.float32)
    arr_flash = np.asarray(flash_spikes, dtype=np.float32)

    jerk = (
        float(np.max(np.abs(np.diff(arr_p99))))
        if len(arr_p99) > 1
        else float(arr_p99[0] * 0.25)
    )
    if len(arr_lap) > 1:
        lap_drops = np.maximum(0.0, -np.diff(arr_lap)) / (arr_lap[:-1] + 1e-5)
        blur_shock = float(np.max(lap_drops))
    else:
        blur_shock = 0.0

    median_p99 = float(np.median(arr_p99)) + 1.0
    motion_ratio = float(np.max(arr_p99)) / median_p99

    return [
        round(min(1.0, float(np.max(arr_mean)) / 40.0), 6),
        round(min(1.0, float(np.max(arr_p99)) / 120.0), 6),
        round(min(1.0, jerk / 35.0), 6),
        round(min(1.0, blur_shock * 10.0), 6),
        round(min(1.0, float(np.max(arr_flash)) / 100.0), 6),
        round(min(1.0, motion_ratio / 4.0), 6),
    ]


def _compute_draw_motion_surge(
    cap: Optional[cv2.VideoCapture], beep_time: float
) -> float:
    """
    Measure shooter draw motion surge after a candidate beep:
    Compares frame motion in [beep_time + 0.15s, beep_time + 0.95s]
    against pre-beep ready stance [beep_time - 0.60s, beep_time - 0.05s].
    Returns a normalized bonus in [0.0, 1.0].
    """
    if cap is None or not cap.isOpened():
        return 0.5

    pre_vec = extract_video_window_features(
        cap, max(0.1, beep_time - 0.30), pre_sec=0.20, post_sec=0.20
    )
    post_vec = extract_video_window_features(
        cap, beep_time + 0.45, pre_sec=0.25, post_sec=0.35
    )
    pre_motion = pre_vec[1] + 0.05
    post_motion = post_vec[1]
    ratio = post_motion / pre_motion
    return float(np.clip((ratio - 0.8) / 2.5, 0.0, 1.0))


def _find_spaced_peaks(
    norm_curve: np.ndarray, min_val: float, min_dist_frames: int, max_peaks: int = 6
) -> List[int]:
    """Iteratively pick peaks >= min_val separated by at least min_dist_frames (handles flat-top plateaus)."""
    work = norm_curve.copy()
    peaks: List[int] = []
    for _ in range(max_peaks):
        idx = int(np.argmax(work))
        if float(work[idx]) < min_val:
            break
        peaks.append(idx)
        lo = max(0, idx - min_dist_frames)
        hi = min(len(work), idx + min_dist_frames + 1)
        work[lo:hi] = 0.0
    return peaks


def detect_multimodal_beeps(
    y: np.ndarray,
    sr: int,
    signature: FirearmAudioSignature,
    cap: Optional[cv2.VideoCapture] = None,
    max_beep_time: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Detect one or two Shot Timer Beeps:
      - If 1 beep is present: returns (start_beep, None)
      - If 2 narrowband timer beeps are present (separated by >= 1.2s):
        returns (first_beep_as_start, second_beep_as_time_limit).
    """
    hop_length = 128
    n_fft_fine = 2048
    S_fine = np.abs(librosa.stft(y, n_fft=n_fft_fine, hop_length=hop_length))
    freqs_fine = librosa.fft_frequencies(sr=sr, n_fft=n_fft_fine)
    flat_fine = librosa.feature.spectral_flatness(
        S=S_fine, n_fft=n_fft_fine, hop_length=hop_length
    )[0]

    max_frame = (
        int(librosa.time_to_frames(max_beep_time, sr=sr, hop_length=hop_length))
        if max_beep_time is not None and max_beep_time > 0.25
        else S_fine.shape[1]
    )
    max_frame = max(10, min(S_fine.shape[1], max_frame))

    timer_mask = (freqs_fine >= 800.0) & (freqs_fine <= 5000.0)
    S_band = S_fine[timer_mask, :max_frame]
    f_band = freqs_fine[timer_mask]
    flat_sub = flat_fine[:max_frame]

    max_bin = np.max(S_band, axis=0)
    mean_bin = np.mean(S_band, axis=0) + 1e-7
    crest = max_bin / mean_bin
    total_e = np.sum(S_fine[:, :max_frame], axis=0) + 1e-7
    e_floor = float(np.percentile(total_e, 25))

    target_freq = signature.beep_profile.frequency_hz or 3000.0
    dom_freq_idx = np.argmax(S_band, axis=0)
    dom_freqs = f_band[dom_freq_idx]
    freq_dist = np.abs(dom_freqs - target_freq)
    freq_affinity = np.exp(-0.5 * (freq_dist / 900.0) ** 2)

    pure_tone_raw = np.where(
        (total_e > e_floor * 1.8) & (crest > 10.0) & (flat_sub < 0.08),
        crest
        * (max_bin / total_e)
        * (1.0 - np.clip(flat_sub, 0.0, 1.0))
        * (0.65 + 0.35 * freq_affinity),
        0.0,
    )

    dur_frames = max(
        5, int(round((signature.beep_profile.duration_ms / 1000.0) * sr / hop_length))
    )
    kernel = np.ones(dur_frames, dtype=np.float32) / float(dur_frames)
    smoothed_crest = np.convolve(pure_tone_raw, kernel, mode="same")
    min_beep_dist_frames = int(0.8 * sr / hop_length)

    if float(np.max(smoothed_crest)) > 1e-3:
        norm_crest = smoothed_crest / float(np.max(smoothed_crest))
        cand_peaks = _find_spaced_peaks(
            norm_crest, min_val=0.35, min_dist_frames=min_beep_dist_frames, max_peaks=6
        )
        if not cand_peaks:
            cand_peaks = [int(np.argmax(smoothed_crest))]

        has_beep_feat = any(abs(x) > 1e-6 for x in signature.beep_profile.features)
        scored_beeps: List[Tuple[float, float, float]] = []

        for pk in cand_peaks[:6]:
            pk_idx = int(pk)
            thresh = 0.25 * smoothed_crest[pk_idx]
            onset_f = pk_idx
            while onset_f > 0 and smoothed_crest[onset_f - 1] >= thresh:
                onset_f -= 1
            b_time = float(
                librosa.frames_to_time(onset_f, sr=sr, hop_length=hop_length)
            )
            c_score = float(norm_crest[pk_idx])

            feat_sim = 0.5
            if has_beep_feat:
                s_i = max(0, int((b_time - 0.02) * sr))
                e_i = min(len(y), int((b_time + 0.35) * sr))
                feat_sim = cosine_similarity(
                    extract_cf032ba_window_features(y[s_i:e_i], sr),
                    signature.beep_profile.features,
                )

            draw_bonus = (
                _compute_draw_motion_surge(cap, b_time) if len(cand_peaks) > 1 else 0.5
            )
            total_b_score = 0.70 * c_score + 0.20 * feat_sim + 0.10 * draw_bonus
            scored_beeps.append((total_b_score, round(b_time, 3), c_score))

        scored_beeps.sort(key=lambda x: x[0], reverse=True)
        primary_time = scored_beeps[0][1]

        # Check if there is a valid second narrowband timer beep >= 1.2s away
        second_time: Optional[float] = None
        for sc, bt, raw_c in scored_beeps[1:]:
            if abs(bt - primary_time) >= 1.20 and raw_c >= 0.35 and sc >= 0.40:
                second_time = bt
                break

        if second_time is not None:
            start_b = min(primary_time, second_time)
            limit_b = max(primary_time, second_time)
            return start_b, limit_b

        return primary_time, None

    # Fallback: Original cf032ba narrowband tonal detector (for synthetic/demo tones)
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))[:, :max_frame]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(
        S=S, n_fft=n_fft, hop_length=hop_length
    )[0]

    beep_band = np.abs(freqs - target_freq) <= 450.0
    if not np.any(beep_band):
        beep_band = (freqs >= 800.0) & (freqs <= 5000.0)

    band_energy = np.sum(S[beep_band, :], axis=0)
    total_energy = np.sum(S, axis=0) + 1e-7
    band_ratio = band_energy / total_energy
    tonal_score = band_energy * band_ratio * (1.0 - np.clip(flatness, 0.0, 1.0))
    smoothed_tonal = np.convolve(tonal_score, kernel, mode="same")

    if float(np.max(smoothed_tonal)) > 1e-4:
        norm_tonal = smoothed_tonal / float(np.max(smoothed_tonal))
        t_peaks = _find_spaced_peaks(
            norm_tonal, min_val=0.35, min_dist_frames=min_beep_dist_frames, max_peaks=6
        )
        if not t_peaks:
            t_peaks = [int(np.argmax(smoothed_tonal))]

        tonal_candidates: List[Tuple[float, float]] = []
        for pk in t_peaks[:6]:
            pk_idx = int(pk)
            if float(flatness[pk_idx]) >= 0.08:
                continue
            threshold = 0.30 * smoothed_tonal[pk_idx]
            onset_frame = pk_idx
            while onset_frame > 0 and smoothed_tonal[onset_frame - 1] >= threshold:
                onset_frame -= 1
            b_t = round(
                float(librosa.frames_to_time(onset_frame, sr=sr, hop_length=hop_length)),
                3,
            )
            tonal_candidates.append((float(norm_tonal[pk_idx]), b_t))

        if tonal_candidates:
            tonal_candidates.sort(key=lambda x: x[0], reverse=True)
            first_t = tonal_candidates[0][1]
            second_t: Optional[float] = None
            for sc, bt in tonal_candidates[1:]:
                if abs(bt - first_t) >= 1.20 and sc >= 0.38:
                    second_t = bt
                    break
            if second_t is not None:
                return min(first_t, second_t), max(first_t, second_t)
            return first_t, None

        peak_idx = int(np.argmax(smoothed_tonal))
        threshold = 0.30 * smoothed_tonal[peak_idx]
        onset_frame = peak_idx
        while onset_frame > 0 and smoothed_tonal[onset_frame - 1] >= threshold:
            onset_frame -= 1
        return (
            round(
                float(librosa.frames_to_time(onset_frame, sr=sr, hop_length=hop_length)),
                3,
            ),
            None,
        )

    return None, None


def detect_multimodal_beep(
    y: np.ndarray,
    sr: int,
    signature: FirearmAudioSignature,
    cap: Optional[cv2.VideoCapture] = None,
    max_beep_time: Optional[float] = None,
) -> Optional[float]:
    start_b, _ = detect_multimodal_beeps(y, sr, signature, cap, max_beep_time)
    return start_b


def propose_candidate_peaks(
    y: np.ndarray,
    sr: int,
    beep_timestamp: Optional[float],
    cap: Optional[cv2.VideoCapture] = None,
    end_beep_timestamp: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Propose candidate transient events after `beep_timestamp` using the original cf032ba
    `rms * (onset_env + 0.5)` envelope and extract each candidate's:
      - 16-D cf032ba audio feature vector
      - 6-D transient & reverb step-up physics features
      - 6-D synchronized OpenCV video recoil/flash features
    """
    hop_length = 128
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    flatness = librosa.feature.spectral_flatness(
        S=S, n_fft=n_fft, hop_length=hop_length
    )[0]

    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    n_frames = min(len(rms), len(onset_env), len(flatness))
    rms = rms[:n_frames]
    onset_env = onset_env[:n_frames]
    flatness = flatness[:n_frames]

    rms_max = float(np.max(rms)) if len(rms) > 0 else 1.0
    onset_max = max(float(np.max(onset_env)) if len(onset_env) > 0 else 1.0, 4.0)

    combined_env = rms * (onset_env + 0.5)
    if float(np.max(combined_env)) > 0:
        combined_env = combined_env / float(np.max(combined_env))

    min_wait_frames = max(1, int(0.11 * sr / hop_length))
    peaks = librosa.util.peak_pick(
        combined_env,
        pre_max=3,
        post_max=3,
        pre_avg=5,
        post_avg=5,
        delta=0.06,
        wait=min_wait_frames,
    )

    min_shot_time = (beep_timestamp + 0.18) if beep_timestamp is not None else 0.05
    hf_mask = (freqs >= 2000.0) & (freqs <= 10000.0)
    lf_mask = (freqs >= 100.0) & (freqs < 1500.0)

    candidates: List[Dict[str, Any]] = []
    for p_frame in peaks:
        p = int(p_frame)
        cand_t = float(librosa.frames_to_time(p, sr=sr, hop_length=hop_length))
        if cand_t < min_shot_time:
            continue

        pre_rms = float(np.mean(rms[max(0, p - 10) : max(1, p - 2)])) / max(
            rms_max, 1e-8
        )
        post_rms = float(np.max(rms[p : min(n_frames, p + 12)])) / max(rms_max, 1e-8)
        if post_rms < 0.12:
            continue

        step_up = post_rms - pre_rms
        loc_onset = float(np.max(onset_env[max(0, p - 2) : min(n_frames, p + 3)]))
        onset_norm = min(1.0, loc_onset / onset_max)
        loc_flat = float(np.max(flatness[max(0, p - 1) : min(n_frames, p + 5)]))

        # Exclude the 2nd (Time Limit) Beep tone itself from being proposed as a gunshot
        if (
            end_beep_timestamp is not None
            and abs(cand_t - end_beep_timestamp) <= 0.25
            and loc_flat < 0.12
        ):
            continue

        # High-frequency muzzle crack ratio in [p, p+4]
        s_slice = S[:, max(0, p - 1) : min(S.shape[1], p + 4)]
        hf_e = float(np.mean(s_slice[hf_mask, :])) if s_slice.shape[1] > 0 else 0.0
        lf_e = float(np.mean(s_slice[lf_mask, :])) if s_slice.shape[1] > 0 else 0.0
        hf_crack_ratio = hf_e / (hf_e + lf_e + 1e-7)

        start_idx = max(0, int((cand_t - 0.015) * sr))
        end_idx = min(len(y), int((cand_t + 0.165) * sr))
        y_win = y[start_idx:end_idx]
        if len(y_win) < 256:
            continue

        audio_16d = extract_cf032ba_window_features(y_win, sr)
        video_6d = extract_video_window_features(cap, cand_t)

        # Refine to exact local waveform peak
        local_start = max(0, int((cand_t - 0.02) * sr))
        local_end = min(len(y), int((cand_t + 0.04) * sr))
        if local_end > local_start:
            local_peak_offset = int(np.argmax(np.abs(y[local_start:local_end])))
            refined_t = round((local_start + local_peak_offset) / float(sr), 3)
        else:
            refined_t = round(cand_t, 3)

        physics_6d = [
            round(post_rms, 6),
            round(pre_rms, 6),
            round(step_up, 6),
            round(onset_norm, 6),
            round(loc_flat, 6),
            round(hf_crack_ratio, 6),
        ]

        candidates.append(
            {
                "cand_time": cand_t,
                "refined_time": refined_t,
                "audio_16d": audio_16d,
                "physics_6d": physics_6d,
                "video_6d": video_6d,
                "post_rms": post_rms,
                "pre_rms": pre_rms,
                "step_up": step_up,
                "loc_onset": loc_onset,
                "loc_flat": loc_flat,
            }
        )

    return candidates


def _build_29d_vector(cand: Dict[str, Any], is_slow_regime: bool) -> List[float]:
    return (
        list(cand["audio_16d"])
        + list(cand["physics_6d"])
        + list(cand["video_6d"])
        + [1.0 if is_slow_regime else 0.0]
    )


def _get_training_file_path(base_dir: Path, gun_name: str) -> Path:
    samples_dir = base_dir / "training_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", gun_name.strip().lower()).strip("_") or "default"
    return samples_dir / f"{slug}.json"


def load_firearm_training_dataset(base_dir: Path, gun_name: str) -> Dict[str, Any]:
    path = _get_training_file_path(base_dir, gun_name)
    if not path.exists():
        return {"gun_name": gun_name, "samples": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"gun_name": gun_name, "samples": []}


def clear_firearm_training_dataset(base_dir: Path, gun_name: str) -> None:
    path = _get_training_file_path(base_dir, gun_name)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def harvest_and_save_training_samples(
    base_dir: Path,
    gun_name: str,
    video_id: str,
    y: np.ndarray,
    sr: int,
    beep_timestamp: Optional[float],
    confirmed_shots: List[float],
    cap: Optional[cv2.VideoCapture] = None,
) -> Tuple[int, int, int, Optional[float]]:
    """
    Harvest Positive shot samples (at confirmed_shots) and Hard Negative samples
    (all rejected candidate peaks in the same video) with full 29-D Audio+Video+Regime vectors.
    Returns (trained_videos_count, positive_shots_count, hard_negatives_count, cv_f1_score).
    """
    dataset = load_firearm_training_dataset(base_dir, gun_name)
    # Remove any old samples from this same video_id so re-saving doesn't duplicate
    existing_samples = [
        s for s in dataset.get("samples", []) if s.get("video_id") != video_id
    ]

    candidates = propose_candidate_peaks(y, sr, beep_timestamp, cap)
    sorted_gt = sorted(float(t) for t in confirmed_shots)

    new_samples: List[Dict[str, Any]] = []
    matched_cand_indices = set()

    # 1. Harvest Positive samples for each confirmed shot
    for idx, gt_t in enumerate(sorted_gt):
        delta_t = (gt_t - sorted_gt[idx - 1]) if idx > 0 else float("inf")
        is_slow = delta_t > SLOW_SHOT_THRESHOLD_SEC

        # Find closest candidate within +-90ms if present
        best_c_idx = None
        best_dist = 0.095
        for c_i, c in enumerate(candidates):
            d = abs(c["refined_time"] - gt_t)
            if d < best_dist:
                best_dist = d
                best_c_idx = c_i

        if best_c_idx is not None:
            matched_cand_indices.add(best_c_idx)
            vec_29d = _build_29d_vector(candidates[best_c_idx], is_slow)
        else:
            # Extract directly at gt_t if it wasn't in candidate peaks
            s_i = max(0, int((gt_t - 0.015) * sr))
            e_i = min(len(y), int((gt_t + 0.165) * sr))
            y_w = y[s_i:e_i] if e_i > s_i else np.zeros(512, dtype=np.float32)
            a_16 = extract_cf032ba_window_features(y_w, sr)
            v_6 = extract_video_window_features(cap, gt_t)
            p_6 = [0.5, 0.1, 0.4, 0.5, 0.3, 0.5]
            vec_29d = a_16 + p_6 + v_6 + [1.0 if is_slow else 0.0]

        new_samples.append(
            {
                "video_id": video_id,
                "timestamp": round(gt_t, 3),
                "label": 1,
                "regime": "slow" if is_slow else "fast",
                "features_29d": vec_29d,
            }
        )

    # 2. Harvest Hard Negative samples (candidates > 140ms away from any confirmed shot)
    for c_i, c in enumerate(candidates):
        if c_i in matched_cand_indices:
            continue
        if any(abs(c["refined_time"] - gt_t) < 0.14 for gt_t in sorted_gt):
            continue

        # Determine what regime this candidate would have been evaluated under
        prior_shots = [gt_t for gt_t in sorted_gt if gt_t < c["refined_time"]]
        delta_prev = (
            (c["refined_time"] - prior_shots[-1]) if prior_shots else float("inf")
        )
        is_slow = delta_prev > SLOW_SHOT_THRESHOLD_SEC
        vec_29d = _build_29d_vector(c, is_slow)

        new_samples.append(
            {
                "video_id": video_id,
                "timestamp": c["refined_time"],
                "label": 0,
                "regime": "slow" if is_slow else "fast",
                "features_29d": vec_29d,
            }
        )

    all_samples = existing_samples + new_samples
    dataset["gun_name"] = gun_name
    dataset["samples"] = all_samples

    path = _get_training_file_path(base_dir, gun_name)
    path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    video_ids = {s.get("video_id") for s in all_samples if s.get("video_id")}
    pos_count = sum(1 for s in all_samples if s.get("label") == 1)
    neg_count = sum(1 for s in all_samples if s.get("label") == 0)
    cv_f1 = _compute_dataset_cv_f1(all_samples)

    return len(video_ids), pos_count, neg_count, cv_f1


def _compute_dataset_cv_f1(samples: List[Dict[str, Any]]) -> Optional[float]:
    """
    Compute Leave-One-Video-Out (or split) cross-validation F1 score once we have
    enough positive and negative examples.
    """
    pos = [s for s in samples if s.get("label") == 1]
    neg = [s for s in samples if s.get("label") == 0]
    if len(pos) < 3 or len(neg) < 2:
        return None

    X = np.asarray([s["features_29d"] for s in samples], dtype=np.float32)
    y_true = np.asarray([int(s["label"]) for s in samples], dtype=np.int32)
    vids = [s.get("video_id", "") for s in samples]
    unique_vids = sorted(set(vids))

    if len(unique_vids) >= 2:
        y_pred = np.zeros_like(y_true)
        for v in unique_vids:
            test_mask = np.asarray([vid == v for vid in vids])
            train_mask = ~test_mask
            if (
                np.sum(train_mask) < 3
                or len(set(y_true[train_mask])) < 2
                or np.sum(test_mask) == 0
            ):
                continue
            clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=5,
                class_weight="balanced",
                random_state=42,
            )
            clf.fit(X[train_mask], y_true[train_mask])
            y_pred[test_mask] = clf.predict(X[test_mask])
        return round(float(f1_score(y_true, y_pred, zero_division=0)), 4)

    # Single-video in-sample fit sanity score
    clf = RandomForestClassifier(
        n_estimators=60, max_depth=5, class_weight="balanced", random_state=42
    )
    clf.fit(X, y_true)
    preds = clf.predict(X)
    return round(float(f1_score(y_true, preds, zero_division=0)), 4)


def detect_multimodal_events_with_limit(
    wav_path: Path,
    signature: FirearmAudioSignature,
    video_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[float], List[float], Optional[float]]:
    """
    Detect Start Beep, Gunshot timestamps, and optional 2nd (Time Limit) Beep.
    Returns (beep_timestamp, shot_timestamps, end_beep_timestamp).
    """
    y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
    if len(y) < sr * 0.2:
        return None, [], None

    cap: Optional[cv2.VideoCapture] = None
    if video_path is not None and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))

    try:
        beep_timestamp, end_beep_timestamp = detect_multimodal_beeps(
            y, sr, signature, cap
        )
        candidates = propose_candidate_peaks(
            y, sr, beep_timestamp, cap, end_beep_timestamp=end_beep_timestamp
        )
        # Safety Net: If beep_timestamp was placed near the end of the video leaving 0 candidates,
        # re-detect the beep prior to the first major broadband transient in the recording
        if not candidates and beep_timestamp is not None:
            all_cands = propose_candidate_peaks(y, sr, None, cap)
            strong_cands = [
                c for c in all_cands if c["post_rms"] >= 0.35 and c["loc_flat"] >= 0.12
            ]
            if strong_cands:
                first_shot_t = strong_cands[0]["cand_time"]
                if beep_timestamp >= first_shot_t - 0.10:
                    earlier_beep, earlier_limit = detect_multimodal_beeps(
                        y, sr, signature, cap, max_beep_time=first_shot_t - 0.15
                    )
                    if earlier_beep is not None:
                        beep_timestamp = earlier_beep
                        end_beep_timestamp = earlier_limit
                        candidates = [
                            c
                            for c in all_cands
                            if c["cand_time"] >= beep_timestamp + 0.18
                        ]
    finally:
        if cap is not None:
            cap.release()

    if not candidates:
        return beep_timestamp, [], end_beep_timestamp

    # Load per-firearm supervised classifier if training samples exist
    clf: Optional[RandomForestClassifier] = None
    storage_root = base_dir or Path(
        os.environ.get("ANALYZER_DATA_DIR", wav_path.parent.parent)
    )
    dataset = load_firearm_training_dataset(storage_root, signature.gun_name)
    samples = dataset.get("samples", [])
    pos_samples = [s for s in samples if s.get("label") == 1]
    neg_samples = [s for s in samples if s.get("label") == 0]

    if len(pos_samples) >= 2 and len(neg_samples) >= 1:
        X_train = np.asarray([s["features_29d"] for s in samples], dtype=np.float32)
        y_train = np.asarray([int(s["label"]) for s in samples], dtype=np.int32)
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)

    sp = signature.shot_profile
    has_ref_features = any(abs(x) > 1e-6 for x in sp.features)
    has_slow_ref = any(abs(x) > 1e-6 for x in sp.slow_features)
    has_fast_ref = any(abs(x) > 1e-6 for x in sp.fast_features)
    has_video_ref = any(abs(x) > 1e-6 for x in sp.video_features)

    shot_timestamps: List[float] = []

    for c in candidates:
        cand_t = c["refined_time"]
        delta_t = (
            (cand_t - shot_timestamps[-1])
            if len(shot_timestamps) > 0
            else float("inf")
        )
        is_slow = delta_t > SLOW_SHOT_THRESHOLD_SEC

        post_rms = c["post_rms"]
        step_up = c["step_up"]
        loc_onset = c["loc_onset"]
        loc_flat = c["loc_flat"]

        # 1. Physics Gate:
        # Reject decaying indoor reverberation tails (where step_up <= 0 or flatness is pure room boom)
        if post_rms < 0.15:
            continue

        if is_slow:
            # Slow shot (>2000ms apart or first shot): room reverb has decayed, so onset & step_up are pronounced
            if step_up < 0.14 or loc_onset < 1.6 or loc_flat < 0.08:
                continue
        else:
            # Fast follow-up shot (<=2000ms apart): riding on top of previous shot's reverb tail!
            # Must show a positive envelope step-up AND broadband muzzle crack flatness
            if step_up < 0.08 or loc_onset < 1.15 or loc_flat < 0.10:
                continue

        # 2. Original cf032ba 16-D Acoustic Template Similarity (Slow vs Fast regime)
        acoustic_sim = 0.6
        if has_ref_features:
            sim_all = cosine_similarity(c["audio_16d"], sp.features)
            if is_slow and has_slow_ref:
                sim_reg = cosine_similarity(c["audio_16d"], sp.slow_features)
            elif not is_slow and has_fast_ref:
                sim_reg = cosine_similarity(c["audio_16d"], sp.fast_features)
            else:
                sim_reg = sim_all
            acoustic_sim = max(sim_all, sim_reg)
            if acoustic_sim < 0.35:
                continue

        # 3. Video Recoil Similarity / Impulse Score
        video_score = 0.5
        if has_video_ref and any(abs(x) > 1e-6 for x in c["video_6d"]):
            v_sim = cosine_similarity(c["video_6d"], sp.video_features)
            v_impulse = max(c["video_6d"][2], c["video_6d"][3], c["video_6d"][1])
            video_score = 0.6 * v_sim + 0.4 * v_impulse

        # 4. Supervised Multimodal Classifier Probability (if trained on positive + hard-negative samples)
        if clf is not None:
            vec_29d = np.asarray(
                [_build_29d_vector(c, is_slow)], dtype=np.float32
            )
            prob_shot = float(clf.predict_proba(vec_29d)[0][1])
            blended_conf = (
                0.55 * prob_shot + 0.35 * acoustic_sim + 0.10 * video_score
            )
            if blended_conf < 0.42:
                continue

        if not shot_timestamps or (cand_t - shot_timestamps[-1]) >= 0.10:
            shot_timestamps.append(round(float(cand_t), 3))

    return beep_timestamp, shot_timestamps, end_beep_timestamp


def detect_multimodal_events(
    wav_path: Path,
    signature: FirearmAudioSignature,
    video_path: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> Tuple[Optional[float], List[float]]:
    beep_t, shots, _ = detect_multimodal_events_with_limit(
        wav_path=wav_path,
        signature=signature,
        video_path=video_path,
        base_dir=base_dir,
    )
    return beep_t, shots


def evaluate_session_predictions(
    gun_name: str,
    signature_version: int,
    ground_truth_beep: Optional[float],
    ground_truth_shots: List[float],
    predicted_beep: Optional[float],
    predicted_shots: List[float],
    tolerance_sec: float = 0.10,
) -> SessionCheckResult:
    """
    Compare automated model predictions against user-verified ground truth markers
    within `tolerance_sec` (+-100ms) and compute Precision, Recall, F1, and Mean Timing Error (ms).
    """
    beep_error_ms: Optional[float] = None
    if ground_truth_beep is not None and predicted_beep is not None:
        beep_error_ms = round(abs(predicted_beep - ground_truth_beep) * 1000.0, 1)

    gt_sorted = sorted(float(t) for t in ground_truth_shots)
    pred_sorted = sorted(float(t) for t in predicted_shots)

    used_pred = set()
    errors_ms: List[float] = []

    for gt_t in gt_sorted:
        best_idx = None
        best_diff = tolerance_sec
        for p_idx, p_t in enumerate(pred_sorted):
            if p_idx in used_pred:
                continue
            diff = abs(p_t - gt_t)
            if diff <= best_diff:
                best_diff = diff
                best_idx = p_idx
        if best_idx is not None:
            used_pred.add(best_idx)
            errors_ms.append(best_diff * 1000.0)

    matched = len(used_pred)
    missed = max(0, len(gt_sorted) - matched)
    fp = max(0, len(pred_sorted) - matched)

    precision = (
        matched / float(len(pred_sorted))
        if len(pred_sorted) > 0
        else (1.0 if len(gt_sorted) == 0 else 0.0)
    )
    recall = (
        matched / float(len(gt_sorted))
        if len(gt_sorted) > 0
        else (1.0 if len(pred_sorted) == 0 else 0.0)
    )
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )
    mean_err = round(float(np.mean(errors_ms)), 1) if errors_ms else None

    return SessionCheckResult(
        gun_name=gun_name,
        signature_version=signature_version,
        beep_error_ms=beep_error_ms,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        mean_shot_error_ms=mean_err,
        matched_shots=matched,
        missed_shots=missed,
        false_positives=fp,
        total_ground_truth_shots=len(gt_sorted),
        total_predicted_shots=len(pred_sorted),
    )
