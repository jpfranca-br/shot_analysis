from typing import List, Optional
from backend.models import AnalyticsSummary, ShotMetricItem


def compute_analytics(
    beep_timestamp: Optional[float],
    shot_timestamps: List[float],
) -> AnalyticsSummary:
    """
    Calculate Draw Time, Split Times, and Total Stage Time (FR-4.1 & FR-4.2).
    """
    sorted_shots = sorted(round(float(t), 4) for t in shot_timestamps if t is not None)
    beep = round(float(beep_timestamp), 4) if beep_timestamp is not None else None

    shot_items: List[ShotMetricItem] = []
    for idx, shot_t in enumerate(sorted_shots):
        shot_num = idx + 1
        elapsed: Optional[float] = None
        split: Optional[float] = None

        if beep is not None:
            elapsed = round(shot_t - beep, 3)

        if idx == 0:
            # First shot split is the draw time (if beep is set)
            split = round(shot_t - beep, 3) if beep is not None else None
        else:
            split = round(shot_t - sorted_shots[idx - 1], 3)

        shot_items.append(
            ShotMetricItem(
                shot_number=shot_num,
                timestamp=shot_t,
                split_time=split,
                elapsed_time=elapsed,
            )
        )

    draw_time: Optional[float] = None
    total_stage_time: Optional[float] = None

    if beep is not None and len(sorted_shots) > 0:
        draw_time = round(sorted_shots[0] - beep, 3)
        total_stage_time = round(sorted_shots[-1] - beep, 3)

    return AnalyticsSummary(
        beep_timestamp=beep,
        draw_time=draw_time,
        total_stage_time=total_stage_time,
        total_shots=len(sorted_shots),
        shots=shot_items,
    )
