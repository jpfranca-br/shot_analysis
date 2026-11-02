from typing import List, Optional
from backend.models import AnalyticsSummary, ShotMetricItem


def compute_analytics(
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
    end_beep_timestamp: Optional[float] = None,
) -> AnalyticsSummary:
    """
    Calculate Draw Time, Split Times, Total Stage Time, and Time Limit (2nd Beep) status (FR-4.1 & FR-4.2).
    """
    sorted_shots = sorted(round(float(t), 4) for t in shot_timestamps if t is not None)
    beep = round(float(beep_timestamp), 4) if beep_timestamp is not None else None
    end_beep = (
        round(float(end_beep_timestamp), 4)
        if end_beep_timestamp is not None
        else None
    )

    # Ensure if both beeps are set, beep is the first (start) and end_beep is the second (time limit)
    if beep is not None and end_beep is not None and end_beep < beep:
        beep, end_beep = end_beep, beep

    shot_items: List[ShotMetricItem] = []
    shots_before = 0
    shots_after = 0

    for idx, shot_t in enumerate(sorted_shots):
        shot_num = idx + 1
        elapsed: Optional[float] = None
        split: Optional[float] = None

        if beep is not None:
            elapsed = round(shot_t - beep, 3)

        if idx == 0:
            split = round(shot_t - beep, 3) if beep is not None else None
        else:
            split = round(shot_t - sorted_shots[idx - 1], 3)

        in_limit = True
        if end_beep is not None:
            in_limit = bool(shot_t <= end_beep)

        if in_limit:
            shots_before += 1
        else:
            shots_after += 1

        shot_items.append(
            ShotMetricItem(
                shot_number=shot_num,
                timestamp=shot_t,
                split_time=split,
                elapsed_time=elapsed,
                within_time_limit=in_limit,
            )
        )

    draw_time: Optional[float] = None
    total_stage_time: Optional[float] = None
    time_limit_dur: Optional[float] = None

    if beep is not None and len(sorted_shots) > 0:
        draw_time = round(sorted_shots[0] - beep, 3)
        total_stage_time = round(sorted_shots[-1] - beep, 3)

    if beep is not None and end_beep is not None:
        time_limit_dur = round(end_beep - beep, 3)

    return AnalyticsSummary(
        beep_timestamp=beep,
        end_beep_timestamp=end_beep,
        time_limit_duration=time_limit_dur,
        draw_time=draw_time,
        total_stage_time=total_stage_time,
        total_shots=len(sorted_shots),
        shots_before_limit=shots_before,
        shots_after_limit=shots_after,
        shots=shot_items,
    )
