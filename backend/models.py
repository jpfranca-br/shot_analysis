from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class BeepProfile(BaseModel):
    frequency_hz: float = Field(..., description="Dominant timer beep frequency in Hz")
    duration_ms: int = Field(..., description="Estimated timer beep duration in milliseconds")
    features: List[float] = Field(default_factory=list, description="Normalized spectral feature vector")


class ShotProfile(BaseModel):
    decay_rate: float = Field(..., description="Envelope log-RMS decay rate (dB/s or slope)")
    features: List[float] = Field(default_factory=list, description="Combined normalized acoustic feature vector")
    isolated_features: List[float] = Field(
        default_factory=list,
        description="Reverb-subtracted acoustic feature vector for isolated / first shots (split > 0.85s)",
    )
    followup_features: List[float] = Field(
        default_factory=list,
        description="Reverb-subtracted acoustic feature vector for rapid follow-up shots (split <= 0.85s)",
    )
    hf_flux_ratio: float = Field(
        default=0.5,
        description="Characteristic high-frequency (1.8-10.5 kHz) positive spectral flux ratio",
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
    shot_timestamps: List[float] = Field(default_factory=list)
    user_modified: bool = False


class ShotMetricItem(BaseModel):
    shot_number: int
    timestamp: float
    split_time: Optional[float] = None
    elapsed_time: Optional[float] = None


class AnalyticsSummary(BaseModel):
    beep_timestamp: Optional[float] = None
    draw_time: Optional[float] = None
    total_stage_time: Optional[float] = None
    total_shots: int = 0
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
    shot_timestamps: List[float] = Field(default_factory=list)
    user_modified: bool = False
    rendered_video_url: Optional[str] = None
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
