from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class BeepProfile(BaseModel):
    frequency_hz: float = Field(..., description="Dominant timer beep frequency in Hz")
    duration_ms: int = Field(..., description="Estimated timer beep duration in milliseconds")
    features: List[float] = Field(default_factory=list, description="Normalized spectral feature vector")


class ShotProfile(BaseModel):
    decay_rate: float = Field(..., description="Envelope log-RMS decay rate (dB/s or slope)")
    features: List[float] = Field(default_factory=list, description="Combined 16-D acoustic feature vector")
    slow_features: List[float] = Field(
        default_factory=list,
        description="16-D acoustic feature vector for slow shots (>2000ms apart or first shot)",
    )
    fast_features: List[float] = Field(
        default_factory=list,
        description="16-D acoustic feature vector for fast follow-up shots (<=2000ms apart)",
    )
    video_features: List[float] = Field(
        default_factory=list,
        description="6-D visual recoil, motion jerk, blur shock, and muzzle flash feature vector",
    )
    trained_videos_count: int = Field(
        default=0,
        description="Number of annotated videos used to train this firearm's multimodal classifier",
    )
    positive_shots_count: int = Field(
        default=0,
        description="Total confirmed positive shot examples harvested for this firearm",
    )
    hard_negatives_count: int = Field(
        default=0,
        description="Total hard negative (reverb/chatter/noise) examples harvested for this firearm",
    )
    cv_f1_score: Optional[float] = Field(
        default=None,
        description="Cross-validation F1 accuracy score across trained videos (0.0 to 1.0)",
    )


class FirearmAudioSignature(BaseModel):
    gun_name: str
    signature_version: int = 1
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    beep_profile: BeepProfile
    shot_profile: ShotProfile


class FirearmCreatePayload(BaseModel):
    gun_name: str
    beep_frequency_hz: float = 3000.0
    beep_duration_ms: int = 220
    shot_decay_rate: float = 120.0


class FirearmUpdatePayload(BaseModel):
    new_gun_name: Optional[str] = None
    beep_frequency_hz: Optional[float] = None
    beep_duration_ms: Optional[int] = None
    shot_decay_rate: Optional[float] = None


class SessionAnnotationPayload(BaseModel):
    video_id: str
    gun_name: str
    beep_timestamp: Optional[float] = None
    end_beep_timestamp: Optional[float] = None
    shot_timestamps: List[float] = Field(default_factory=list)
    user_modified: bool = False


class SessionCheckPayload(BaseModel):
    video_id: str
    gun_name: str
    ground_truth_beep: Optional[float] = None
    ground_truth_end_beep: Optional[float] = None
    ground_truth_shots: List[float] = Field(default_factory=list)
    predicted_beep: Optional[float] = None
    predicted_end_beep: Optional[float] = None
    predicted_shots: Optional[List[float]] = None


class SessionCheckResult(BaseModel):
    gun_name: str
    signature_version: int
    beep_error_ms: Optional[float] = None
    end_beep_error_ms: Optional[float] = None
    precision: float
    recall: float
    f1_score: float
    mean_shot_error_ms: Optional[float] = None
    matched_shots: int
    missed_shots: int
    false_positives: int
    total_ground_truth_shots: int
    total_predicted_shots: int


class ShotMetricItem(BaseModel):
    shot_number: int
    timestamp: float
    split_time: Optional[float] = None
    elapsed_time: Optional[float] = None
    within_time_limit: bool = True


class AnalyticsSummary(BaseModel):
    beep_timestamp: Optional[float] = None
    end_beep_timestamp: Optional[float] = None
    time_limit_duration: Optional[float] = None
    draw_time: Optional[float] = None
    total_stage_time: Optional[float] = None
    total_shots: int = 0
    shots_before_limit: int = 0
    shots_after_limit: int = 0
    shots: List[ShotMetricItem] = Field(default_factory=list)


class SessionRecord(BaseModel):
    video_id: str
    filename: str
    gun_name: str
    video_url: str
    audio_url: str
    duration_sec: float
    mode: str  # "training" or "automated"
    signature_version: Optional[int] = None
    beep_timestamp: Optional[float] = None
    end_beep_timestamp: Optional[float] = None
    shot_timestamps: List[float] = Field(default_factory=list)
    predicted_beep: Optional[float] = None
    predicted_end_beep: Optional[float] = None
    predicted_shots: List[float] = Field(default_factory=list)
    user_modified: bool = False
    rendered_video_url: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
