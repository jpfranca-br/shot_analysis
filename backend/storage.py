from datetime import datetime, timezone
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from backend.models import (
    BeepProfile,
    FirearmAudioSignature,
    SessionRecord,
    ShotProfile,
)

DEFAULT_FIREARMS: List[str] = [
    "CZ P-10C 9mm",
    "T4 5.56x45mm",
    "CBC 7022 .22LR",
]


def _make_untrained_signature(
    gun_name: str,
    beep_frequency_hz: float = 3000.0,
    beep_duration_ms: int = 220,
    shot_decay_rate: float = 120.0,
) -> FirearmAudioSignature:
    return FirearmAudioSignature(
        gun_name=gun_name.strip(),
        signature_version=0,
        updated_at=datetime.now(timezone.utc).isoformat(),
        beep_profile=BeepProfile(
            frequency_hz=round(float(beep_frequency_hz), 2),
            duration_ms=int(beep_duration_ms),
            features=[0.0] * 16,
        ),
        shot_profile=ShotProfile(
            decay_rate=round(float(shot_decay_rate), 3),
            features=[0.0] * 16,
            isolated_features=[0.0] * 16,
            followup_features=[0.0] * 16,
            hf_flux_ratio=0.5,
        ),
    )


class StorageManager:
    """Thread-safe local JSON storage for firearm audio signatures and session metadata."""

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(
            base_dir
            or os.environ.get(
                "ANALYZER_DATA_DIR",
                Path(__file__).resolve().parent.parent / "data",
            )
        )
        self.uploads_dir = self.base_dir / "uploads"
        self.outputs_dir = self.base_dir / "outputs"
        self.signatures_path = self.base_dir / "signatures.json"
        self.sessions_path = self.base_dir / "sessions.json"
        self._lock = threading.RLock()
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        if not self.signatures_path.exists():
            initial = {
                self._normalize_gun_key(name): _make_untrained_signature(name).model_dump()
                for name in DEFAULT_FIREARMS
            }
            self.signatures_path.write_text(
                json.dumps(initial, indent=2), encoding="utf-8"
            )
        if not self.sessions_path.exists():
            self.sessions_path.write_text("{}", encoding="utf-8")

    @staticmethod
    def _normalize_gun_key(gun_name: str) -> str:
        return gun_name.strip().lower()

    def list_signatures(self) -> List[FirearmAudioSignature]:
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            return [FirearmAudioSignature(**val) for val in raw.values()]

    def get_signature(
        self, gun_name: str, include_untrained: bool = False
    ) -> Optional[FirearmAudioSignature]:
        key = self._normalize_gun_key(gun_name)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            entry = raw.get(key)
            if not entry:
                return None
            sig = FirearmAudioSignature(**entry)
            if not include_untrained and sig.signature_version <= 0:
                return None
            return sig

    def save_signature(self, signature: FirearmAudioSignature) -> FirearmAudioSignature:
        key = self._normalize_gun_key(signature.gun_name)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            raw[key] = signature.model_dump()
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return signature

    def create_firearm(
        self,
        gun_name: str,
        beep_frequency_hz: float = 3000.0,
        beep_duration_ms: int = 220,
        shot_decay_rate: float = 120.0,
    ) -> FirearmAudioSignature:
        cleaned = gun_name.strip()
        key = self._normalize_gun_key(cleaned)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            if key in raw:
                return FirearmAudioSignature(**raw[key])
            sig = _make_untrained_signature(
                cleaned, beep_frequency_hz, beep_duration_ms, shot_decay_rate
            )
            raw[key] = sig.model_dump()
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return sig

    def update_firearm(
        self,
        gun_name: str,
        new_gun_name: Optional[str] = None,
        beep_frequency_hz: Optional[float] = None,
        beep_duration_ms: Optional[int] = None,
        shot_decay_rate: Optional[float] = None,
    ) -> Optional[FirearmAudioSignature]:
        old_key = self._normalize_gun_key(gun_name)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            if old_key not in raw:
                return None

            sig = FirearmAudioSignature(**raw[old_key])
            target_name = (
                new_gun_name.strip()
                if new_gun_name and new_gun_name.strip()
                else sig.gun_name
            )
            new_key = self._normalize_gun_key(target_name)

            if new_key != old_key:
                del raw[old_key]
            sig.gun_name = target_name

            if beep_frequency_hz is not None:
                sig.beep_profile.frequency_hz = round(float(beep_frequency_hz), 2)
            if beep_duration_ms is not None:
                sig.beep_profile.duration_ms = int(beep_duration_ms)
            if shot_decay_rate is not None:
                sig.shot_profile.decay_rate = round(float(shot_decay_rate), 3)

            sig.updated_at = datetime.now(timezone.utc).isoformat()
            raw[new_key] = sig.model_dump()
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return sig

    def reset_signature(self, gun_name: str) -> Optional[FirearmAudioSignature]:
        key = self._normalize_gun_key(gun_name)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            if key not in raw:
                return None
            existing = FirearmAudioSignature(**raw[key])
            reset_sig = _make_untrained_signature(existing.gun_name)
            raw[key] = reset_sig.model_dump()
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return reset_sig

    def delete_signature(self, gun_name: str) -> bool:
        key = self._normalize_gun_key(gun_name)
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            if key not in raw:
                return False
            del raw[key]
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return True

    def restore_default_firearms(self) -> List[FirearmAudioSignature]:
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.signatures_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            for name in DEFAULT_FIREARMS:
                k = self._normalize_gun_key(name)
                if k not in raw:
                    raw[k] = _make_untrained_signature(name).model_dump()
            self.signatures_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return [FirearmAudioSignature(**val) for val in raw.values()]

    def save_session(self, session: SessionRecord) -> SessionRecord:
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            raw[session.video_id] = session.model_dump()
            self.sessions_path.write_text(
                json.dumps(raw, indent=2), encoding="utf-8"
            )
            return session

    def get_session(self, video_id: str) -> Optional[SessionRecord]:
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            entry = raw.get(video_id)
            if not entry:
                return None
            return SessionRecord(**entry)

    def list_sessions(self) -> List[SessionRecord]:
        with self._lock:
            self.ensure_dirs()
            try:
                raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw = {}
            return [SessionRecord(**val) for val in raw.values()]

    def cleanup_videos(
        self,
        keep_uploaded: bool = False,
        keep_rendered: bool = False,
        exclude_video_id: Optional[str] = None,
    ) -> int:
        """
        Delete uploaded and/or rendered video files when storage is disabled in config.json.
        Preserves firearm audio signatures (signatures.json).
        Returns the number of deleted files.
        """
        deleted_count = 0
        with self._lock:
            self.ensure_dirs()
            if not keep_uploaded:
                for p in self.uploads_dir.iterdir():
                    if not p.is_file() or p.name == ".gitkeep":
                        continue
                    if exclude_video_id and p.name.startswith(exclude_video_id):
                        continue
                    try:
                        p.unlink(missing_ok=True)
                        deleted_count += 1
                    except OSError:
                        pass

            if not keep_rendered:
                for p in self.outputs_dir.iterdir():
                    if not p.is_file() or p.name == ".gitkeep":
                        continue
                    if exclude_video_id and p.name.startswith(exclude_video_id):
                        continue
                    try:
                        p.unlink(missing_ok=True)
                        deleted_count += 1
                    except OSError:
                        pass

            if not keep_uploaded and not keep_rendered:
                try:
                    raw = json.loads(self.sessions_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    raw = {}
                if exclude_video_id and exclude_video_id in raw:
                    new_raw = {exclude_video_id: raw[exclude_video_id]}
                else:
                    new_raw = {}
                self.sessions_path.write_text(
                    json.dumps(new_raw, indent=2), encoding="utf-8"
                )

        return deleted_count

