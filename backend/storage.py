import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from backend.models import FirearmAudioSignature, SessionRecord


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
            self.signatures_path.write_text("{}", encoding="utf-8")
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

    def get_signature(self, gun_name: str) -> Optional[FirearmAudioSignature]:
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
            return FirearmAudioSignature(**entry)

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

