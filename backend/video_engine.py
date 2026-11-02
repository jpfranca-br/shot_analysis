from pathlib import Path
import subprocess
from typing import List, Optional

import cv2
import imageio_ffmpeg
import numpy as np

from backend.analytics import compute_analytics
from backend.models import ShotMetricItem


def _draw_rounded_panel(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bg_color: tuple,
    border_color: tuple,
    alpha: float = 0.78,
    border_thickness: int = 2,
) -> None:
    """Draw a semi-transparent tactical HUD panel on the frame."""
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = img[y1:y2, x1:x2]
    overlay = np.full_like(roi, bg_color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), border_color, border_thickness, cv2.LINE_AA)


def render_annotated_video(
    input_video_path: Path,
    output_video_path: Path,
    gun_name: str,
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
    end_beep_timestamp: Optional[float] = None,
    popup_duration_sec: float = 1.6,
) -> Path:
    """
    Generate an annotated output video using opencv-python (FR-5.1, FR-5.2, FR-5.3).
    Displays:
    - A clear visual indicator when the Start Timer Beep occurs (t = 0.00s) and optional Time Limit (2nd) Beep.
    - A pop-up and persistent text overlay showing Shot Number, Split Time,
      and Total Elapsed Time, color-coded for shots before vs. after the Time Limit.
    """
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    analytics = compute_analytics(
        beep_timestamp, shot_timestamps, end_beep_timestamp=end_beep_timestamp
    )
    beep_timestamp = analytics.beep_timestamp
    end_beep_timestamp = analytics.end_beep_timestamp
    shots: List[ShotMetricItem] = analytics.shots

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 240.0:
        fps = 30.0
    # Use a clean constant output frame rate (e.g. 30.0 fps for ~29-30fps VFR clips)
    out_fps = 30.0 if 23.0 <= fps <= 31.0 else round(float(fps), 2)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    temp_raw_mp4 = output_video_path.with_suffix(".cv2_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_raw_mp4), fourcc, out_fps, (width, height))

    font = cv2.FONT_HERSHEY_DUPLEX
    scale_factor = max(0.55, min(1.3, width / 1000.0))

    src_frame_idx = 0
    written_frames = 0
    prev_pts = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Read true container presentation timestamp (handles Variable Frame Rate phone videos)
        pts_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if src_frame_idx == 0:
            t = 0.0
        elif pts_msec and pts_msec > 0:
            t = pts_msec / 1000.0
            if t <= prev_pts:
                t = prev_pts + (1.0 / out_fps)
        else:
            t = prev_pts + (1.0 / out_fps)

        prev_pts = t
        src_frame_idx += 1

        # 1. Persistent Top-Left Header Bar (Firearm & Stage Timer)
        pad = int(16 * scale_factor)
        header_w = int(370 * scale_factor)
        header_h = int(76 * scale_factor)
        _draw_rounded_panel(
            frame,
            pad,
            pad,
            pad + header_w,
            pad + header_h,
            bg_color=(20, 24, 32),
            border_color=(80, 180, 255),
            alpha=0.78,
            border_thickness=2,
        )

        cv2.putText(
            frame,
            f"FIREARM: {gun_name.upper()}",
            (pad + int(14 * scale_factor), pad + int(28 * scale_factor)),
            font,
            0.55 * scale_factor,
            (190, 215, 255),
            1,
            cv2.LINE_AA,
        )

        if beep_timestamp is not None:
            rel_t = t - beep_timestamp
            timer_str = (
                f"STAGE TIME: {rel_t:+.2f}s"
                if rel_t < 0
                else f"STAGE TIME: {rel_t:.2f}s"
            )
            timer_color = (140, 160, 180) if rel_t < 0 else (90, 240, 175)
        else:
            timer_str = f"VIDEO TIME: {t:.2f}s"
            timer_color = (220, 220, 220)

        cv2.putText(
            frame,
            timer_str,
            (pad + int(14 * scale_factor), pad + int(58 * scale_factor)),
            font,
            0.65 * scale_factor,
            timer_color,
            2,
            cv2.LINE_AA,
        )

        # 2. Timer Beep Visual Indicator (FR-5.2: t = 0.00s indicator)
        if beep_timestamp is not None and beep_timestamp <= t <= beep_timestamp + 1.35:
            beep_age = t - beep_timestamp
            beep_w = int(340 * scale_factor)
            beep_h = int(58 * scale_factor)
            bx1 = max(pad, (width - beep_w) // 2)
            by1 = pad
            _draw_rounded_panel(
                frame,
                bx1,
                by1,
                bx1 + beep_w,
                by1 + beep_h,
                bg_color=(12, 65, 95),
                border_color=(0, 215, 255),
                alpha=0.88,
                border_thickness=2,
            )
            # Pulsing indicator dot
            dot_radius = int((8 + 4 * np.sin(beep_age * 16.0)) * scale_factor)
            cv2.circle(
                frame,
                (bx1 + int(26 * scale_factor), by1 + beep_h // 2),
                max(4, dot_radius),
                (0, 225, 255),
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "TIMER BEEP  |  t = 0.00s",
                (bx1 + int(48 * scale_factor), by1 + int(36 * scale_factor)),
                font,
                0.62 * scale_factor,
                (180, 245, 255),
                2,
                cv2.LINE_AA,
            )

        # 2b. Second Beep (Time Limit) Visual Indicator
        if (
            end_beep_timestamp is not None
            and end_beep_timestamp <= t <= end_beep_timestamp + 1.35
        ):
            limit_age = t - end_beep_timestamp
            beep_w = int(380 * scale_factor)
            beep_h = int(58 * scale_factor)
            bx1 = max(pad, (width - beep_w) // 2)
            by1 = pad
            _draw_rounded_panel(
                frame,
                bx1,
                by1,
                bx1 + beep_w,
                by1 + beep_h,
                bg_color=(65, 18, 78),
                border_color=(245, 95, 215),
                alpha=0.90,
                border_thickness=2,
            )
            dot_radius = int((8 + 4 * np.sin(limit_age * 16.0)) * scale_factor)
            cv2.circle(
                frame,
                (bx1 + int(26 * scale_factor), by1 + beep_h // 2),
                max(4, dot_radius),
                (250, 110, 230),
                -1,
                cv2.LINE_AA,
            )
            limit_rel = (
                f"+{end_beep_timestamp - beep_timestamp:.2f}s"
                if beep_timestamp is not None
                else f"{end_beep_timestamp:.2f}s"
            )
            cv2.putText(
                frame,
                f"TIME LIMIT BEEP  |  {limit_rel}",
                (bx1 + int(48 * scale_factor), by1 + int(36 * scale_factor)),
                font,
                0.60 * scale_factor,
                (245, 200, 255),
                2,
                cv2.LINE_AA,
            )

        # 3. Shot Event Pop-up Overlay (FR-5.2: 1-2s following each shot event)
        active_shot: Optional[ShotMetricItem] = None
        for s in shots:
            if s.timestamp <= t <= s.timestamp + popup_duration_sec:
                active_shot = s

        if active_shot is not None:
            card_w = int(450 * scale_factor)
            card_h = int(112 * scale_factor)
            cx1 = pad
            cy1 = height - card_h - pad

            has_limit = end_beep_timestamp is not None
            in_limit = active_shot.within_time_limit
            if has_limit:
                card_border = (115, 225, 45) if in_limit else (35, 115, 255)
                title_color = (120, 250, 95) if in_limit else (70, 145, 255)
                limit_tag = " [IN LIMIT]" if in_limit else " [AFTER LIMIT]"
            else:
                card_border = (80, 120, 255)
                title_color = (90, 210, 255)
                limit_tag = ""

            _draw_rounded_panel(
                frame,
                cx1,
                cy1,
                cx1 + card_w,
                cy1 + card_h,
                bg_color=(18, 22, 34),
                border_color=card_border,
                alpha=0.86,
                border_thickness=2,
            )

            split_label = (
                f"DRAW: {active_shot.split_time:.2f}s"
                if active_shot.shot_number == 1 and active_shot.split_time is not None
                else (
                    f"SPLIT: +{active_shot.split_time:.2f}s"
                    if active_shot.split_time is not None
                    else "SPLIT: N/A"
                )
            )
            elapsed_label = (
                f"TOTAL ELAPSED: {active_shot.elapsed_time:.2f}s{limit_tag}"
                if active_shot.elapsed_time is not None
                else f"TIME: {active_shot.timestamp:.2f}s{limit_tag}"
            )

            cv2.putText(
                frame,
                f"SHOT #{active_shot.shot_number}",
                (cx1 + int(16 * scale_factor), cy1 + int(34 * scale_factor)),
                font,
                0.78 * scale_factor,
                title_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                split_label,
                (cx1 + int(175 * scale_factor), cy1 + int(34 * scale_factor)),
                font,
                0.70 * scale_factor,
                (110, 250, 175),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                elapsed_label,
                (cx1 + int(16 * scale_factor), cy1 + int(78 * scale_factor)),
                font,
                0.68 * scale_factor,
                (240, 245, 255),
                2,
                cv2.LINE_AA,
            )

        # 4. Persistent Shot Split Ticker on Right Side for Fired Shots
        fired_shots = [s for s in shots if s.timestamp <= t]
        if fired_shots:
            visible_shots = fired_shots[-6:]  # Show up to last 6 shots
            row_h = int(28 * scale_factor)
            panel_w = int(290 * scale_factor)
            panel_h = int(42 * scale_factor) + len(visible_shots) * row_h
            rx1 = width - panel_w - pad
            ry1 = pad
            _draw_rounded_panel(
                frame,
                rx1,
                ry1,
                rx1 + panel_w,
                ry1 + panel_h,
                bg_color=(16, 20, 28),
                border_color=(90, 110, 140),
                alpha=0.75,
                border_thickness=1,
            )
            cv2.putText(
                frame,
                "SHOT LOG (SPLIT / TOTAL)",
                (rx1 + int(12 * scale_factor), ry1 + int(24 * scale_factor)),
                font,
                0.46 * scale_factor,
                (165, 190, 220),
                1,
                cv2.LINE_AA,
            )
            for idx, s in enumerate(visible_shots):
                ry = ry1 + int(48 * scale_factor) + idx * row_h
                sp_txt = f"+{s.split_time:.2f}s" if s.split_time is not None else "--"
                el_txt = f"{s.elapsed_time:.2f}s" if s.elapsed_time is not None else f"{s.timestamp:.2f}s"
                line_txt = f"#{s.shot_number:<2}  {sp_txt:<8}  {el_txt}"
                is_latest = s.shot_number == fired_shots[-1].shot_number
                if end_beep_timestamp is not None:
                    if s.within_time_limit:
                        color = (100, 255, 140) if is_latest else (150, 230, 170)
                    else:
                        color = (60, 130, 255) if is_latest else (95, 155, 250)
                else:
                    color = (100, 255, 180) if is_latest else (220, 225, 235)
                cv2.putText(
                    frame,
                    line_txt,
                    (rx1 + int(12 * scale_factor), ry),
                    font,
                    0.48 * scale_factor,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        # Write frame(s) to CFR output writer so output timestamp matches source VFR PTS
        target_cfr_idx = int(round(t * out_fps))
        repeats = max(1, min(15, target_cfr_idx - written_frames + 1))
        for _ in range(repeats):
            writer.write(frame)
            written_frames += 1

    cap.release()
    writer.release()

    # Remux with original audio (aligned to first_pts=0) and transcode to H.264 MP4
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(temp_raw_mp4),
        "-i",
        str(input_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        "-af",
        "aresample=async=1:first_pts=0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not output_video_path.exists():
        # Fallback: keep the OpenCV MP4 output directly if ffmpeg mux fails
        if temp_raw_mp4.exists():
            temp_raw_mp4.replace(output_video_path)
    else:
        try:
            temp_raw_mp4.unlink(missing_ok=True)
        except OSError:
            pass

    return output_video_path
