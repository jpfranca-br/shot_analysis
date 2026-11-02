from pathlib import Path
import subprocess
from typing import List, Optional

import cv2
import imageio_ffmpeg
import numpy as np
import soundfile as sf


def generate_synthetic_practice_video(
    output_mp4_path: Path,
    duration_sec: float = 5.0,
    fps: float = 30.0,
    beep_time: float = 1.00,
    shot_times: List[float] = None,
    beep_freq_hz: float = 3000.0,
    end_beep_time: Optional[float] = None,
) -> Path:
    """
    Generate a realistic synthetic shooting practice video (.mp4) with:
    - Visual range target, stage clock, and muzzle flash at each shot
    - Acoustic 22050 Hz track containing a clear timer beep (plus optional 2nd time-limit beep) and gunshot transients
    """
    if shot_times is None:
        shot_times = [2.25, 2.70, 3.10, 3.55]

    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    sr = 22050
    total_samples = int(duration_sec * sr)
    t_audio = np.arange(total_samples, dtype=np.float32) / float(sr)

    # Low ambient range noise
    rng = np.random.default_rng(42)
    audio = rng.normal(0.0, 0.004, size=total_samples).astype(np.float32)

    # 1. Add Timer Beep(s) (pure tone + slight harmonic, 220 ms duration)
    beep_dur = 0.22
    beep_list = [beep_time]
    if end_beep_time is not None:
        beep_list.append(end_beep_time)
    for b_time_val in beep_list:
        beep_start = int(b_time_val * sr)
        beep_end = min(total_samples, int((b_time_val + beep_dur) * sr))
        if beep_end > beep_start:
            bt = np.arange(beep_end - beep_start, dtype=np.float32) / float(sr)
            envelope = np.ones_like(bt)
            fade = min(len(bt) // 6, int(0.015 * sr))
            if fade > 0:
                envelope[:fade] = np.linspace(0.0, 1.0, fade)
                envelope[-fade:] = np.linspace(1.0, 0.0, fade)
            tone = (
                0.55 * np.sin(2.0 * np.pi * beep_freq_hz * bt)
                + 0.12 * np.sin(2.0 * np.pi * (beep_freq_hz * 2.0) * bt)
            )
            audio[beep_start:beep_end] += tone * envelope

    # 2. Add Gunshot Impulses (broadband burst + low-mid punch with exponential decay)
    for st in shot_times:
        s_start = int(st * sr)
        s_dur = 0.16
        s_end = min(total_samples, int((st + s_dur) * sr))
        if s_end > s_start:
            st_vec = np.arange(s_end - s_start, dtype=np.float32) / float(sr)
            decay_env = np.exp(-st_vec * 28.0)
            burst_noise = rng.normal(0.0, 1.0, size=len(st_vec)).astype(np.float32)
            punch = 0.65 * np.sin(2.0 * np.pi * 190.0 * st_vec) * np.exp(-st_vec * 45.0)
            crack = 0.85 * burst_noise * decay_env
            shot_wave = punch + crack
            audio[s_start:s_end] += shot_wave

    peak = float(np.max(np.abs(audio)))
    if peak > 1e-6:
        audio = (audio / peak) * 0.92

    temp_wav = output_mp4_path.with_suffix(".synth.wav")
    temp_vid = output_mp4_path.with_suffix(".synth_vid.mp4")
    sf.write(str(temp_wav), audio, sr, subtype="PCM_16")

    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_vid), fourcc, fps, (width, height))
    total_frames = int(duration_sec * fps)

    for f_idx in range(total_frames):
        t = f_idx / fps
        frame = np.full((height, width, 3), (28, 33, 38), dtype=np.uint8)

        # Draw range background grid & silhouette target
        cv2.line(frame, (0, 380), (width, 380), (50, 60, 70), 2)
        target_cx, target_cy = width // 2, 230
        cv2.rectangle(
            frame,
            (target_cx - 70, target_cy - 90),
            (target_cx + 70, target_cy + 90),
            (145, 175, 200),
            -1,
        )
        cv2.rectangle(
            frame,
            (target_cx - 35, target_cy - 55),
            (target_cx + 35, target_cy + 55),
            (115, 145, 170),
            2,
        )

        # Draw bullet impacts that have occurred up to frame time t
        impact_offsets = [(-15, -18), (12, -8), (-6, 15), (18, 20), (0, -30)]
        for idx, st in enumerate(shot_times):
            if t >= st:
                ox, oy = impact_offsets[idx % len(impact_offsets)]
                cv2.circle(
                    frame,
                    (target_cx + ox, target_cy + oy),
                    6,
                    (20, 20, 25),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (target_cx + ox, target_cy + oy),
                    8,
                    (40, 190, 255),
                    1,
                    cv2.LINE_AA,
                )
            if st <= t <= st + 0.08:
                # Muzzle flash visual cue
                cv2.circle(
                    frame,
                    (width // 2, height - 85),
                    55,
                    (80, 210, 255),
                    -1,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    writer.release()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(temp_vid),
        "-i",
        str(temp_wav),
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(output_mp4_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    try:
        temp_wav.unlink(missing_ok=True)
        temp_vid.unlink(missing_ok=True)
    except OSError:
        pass

    return output_mp4_path
