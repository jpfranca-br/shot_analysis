import time
from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import pytest
import soundfile as sf

from backend.demo_generator import generate_synthetic_practice_video
from backend.main import app, get_storage
from backend.storage import StorageManager


@pytest.fixture()
def temp_storage(tmp_path: Path) -> StorageManager:
    storage = StorageManager(base_dir=tmp_path / "test_data")
    app.dependency_overrides[get_storage] = lambda: storage
    yield storage
    app.dependency_overrides.clear()


@pytest.fixture()
def client(temp_storage: StorageManager) -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    vid_dir = tmp_path_factory.mktemp("video_fixtures")
    mp4_path = vid_dir / "test_drill.mp4"
    generate_synthetic_practice_video(
        output_mp4_path=mp4_path,
        duration_sec=4.2,
        fps=30.0,
        beep_time=1.00,
        shot_times=[2.25, 2.70, 3.10],
        beep_freq_hz=3000.0,
    )
    return mp4_path


def test_upload_validation(client: TestClient, tmp_path: Path):
    """Verify FR-1.1 and FR-1.2 validation for unsupported formats and empty firearm names."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("not a video", encoding="utf-8")

    with bad_file.open("rb") as f:
        res = client.post(
            "/api/upload",
            files={"video": ("bad.txt", f, "text/plain")},
            data={"gun_name": "Glock 19"},
        )
    assert res.status_code == 400
    assert "Unsupported video format" in res.json()["detail"]

    with bad_file.open("rb") as f:
        res_empty_gun = client.post(
            "/api/upload",
            files={"video": ("clip.mp4", f, "video/mp4")},
            data={"gun_name": "   "},
        )
    assert res_empty_gun.status_code == 400


def test_full_workflow_cold_start_learning_automated_and_rendering(
    client: TestClient,
    temp_storage: StorageManager,
    sample_video_path: Path,
):
    """
    End-to-end verification of Modules 1 through 5:
    1. Cold Start upload & normalized mono PCM extraction (FR-1.3, FR-2.2)
    2. Manual annotation confirmation & profile creation (FR-2.4, FR-4.1)
    3. Automated peak detection & template matching on second upload (FR-2.3 & NFR < 5s)
    4. Profile refinement on marker edit (FR-2.4)
    5. OpenCV video overlay rendering & download (FR-5.1, FR-5.2, FR-5.3)
    """
    # Step 1: Cold Start Upload for "Glock 19"
    with sample_video_path.open("rb") as vf:
        res = client.post(
            "/api/upload",
            files={"video": ("drill_1.mp4", vf, "video/mp4")},
            data={"gun_name": "Glock 19"},
        )
    assert res.status_code == 200
    data = res.json()
    session_1 = data["session"]
    video_id_1 = session_1["video_id"]

    assert session_1["mode"] == "training"
    assert session_1["beep_timestamp"] is None
    assert session_1["shot_timestamps"] == []
    assert data["signature"] is None

    # Verify normalized mono PCM WAV file was extracted (FR-1.3)
    wav_path = temp_storage.uploads_dir / f"{video_id_1}.wav"
    assert wav_path.exists()
    info = sf.info(str(wav_path))
    assert info.channels == 1
    assert info.subtype == "PCM_16"

    # Step 2: Confirm annotations (Beep = 1.00s, Shots = [2.25, 2.70, 3.10])
    annotation_payload = {
        "video_id": video_id_1,
        "gun_name": "Glock 19",
        "beep_timestamp": 1.00,
        "shot_timestamps": [2.25, 2.70, 3.10],
        "user_modified": True,
    }
    res_ann = client.post("/api/annotations", json=annotation_payload)
    assert res_ann.status_code == 200
    ann_data = res_ann.json()

    # Check Section 6 Firearm Audio Signature Schema
    sig_v1 = ann_data["signature"]
    assert sig_v1["gun_name"] == "Glock 19"
    assert sig_v1["signature_version"] == 1
    assert "updated_at" in sig_v1
    assert 2600.0 <= sig_v1["beep_profile"]["frequency_hz"] <= 3400.0
    assert sig_v1["beep_profile"]["duration_ms"] > 0
    assert len(sig_v1["beep_profile"]["features"]) == 16
    assert sig_v1["shot_profile"]["decay_rate"] > 0
    assert len(sig_v1["shot_profile"]["features"]) == 16

    # Check Module 4 Analytics & Split Calculation (FR-4.1)
    analytics = ann_data["analytics"]
    assert analytics["draw_time"] == pytest.approx(1.25, abs=1e-3)
    assert analytics["total_stage_time"] == pytest.approx(2.10, abs=1e-3)
    assert analytics["total_shots"] == 3
    assert analytics["shots"][0]["split_time"] == pytest.approx(1.25, abs=1e-3)
    assert analytics["shots"][1]["split_time"] == pytest.approx(0.45, abs=1e-3)
    assert analytics["shots"][2]["split_time"] == pytest.approx(0.40, abs=1e-3)
    assert analytics["shots"][2]["elapsed_time"] == pytest.approx(2.10, abs=1e-3)

    # Step 3: Second upload for "Glock 19" triggers Automated Detection (FR-2.3)
    t0 = time.perf_counter()
    with sample_video_path.open("rb") as vf:
        res_auto = client.post(
            "/api/upload",
            files={"video": ("drill_2.mp4", vf, "video/mp4")},
            data={"gun_name": "Glock 19"},
        )
    elapsed_upload = time.perf_counter() - t0
    assert res_auto.status_code == 200
    # NFR Performance check: must complete well within 5 seconds
    assert elapsed_upload < 5.0

    auto_data = res_auto.json()
    session_2 = auto_data["session"]
    assert session_2["mode"] == "automated"
    assert session_2["signature_version"] == 1
    assert session_2["beep_timestamp"] is not None
    assert abs(session_2["beep_timestamp"] - 1.00) <= 0.12
    assert len(session_2["shot_timestamps"]) == 3
    for predicted, expected in zip(session_2["shot_timestamps"], [2.25, 2.70, 3.10]):
        assert abs(predicted - expected) <= 0.08

    # Step 4: Refine profile & render annotated video (FR-2.4 & FR-5.1 - FR-5.3)
    render_payload = {
        "video_id": session_2["video_id"],
        "gun_name": "Glock 19",
        "beep_timestamp": 1.00,
        "shot_timestamps": [2.25, 2.70, 3.10],
        "user_modified": True,
    }
    res_render = client.post("/api/render", json=render_payload)
    assert res_render.status_code == 200
    render_data = res_render.json()

    # Profile version incremented to 2
    assert render_data["signature"]["signature_version"] == 2
    download_url = render_data["download_url"]
    assert download_url == f"/api/videos/{session_2['video_id']}/download"

    # Verify rendered video file exists and can be opened by OpenCV
    rendered_path = temp_storage.outputs_dir / f"{session_2['video_id']}_annotated.mp4"
    assert rendered_path.exists()
    assert rendered_path.stat().st_size > 1000

    cap = cv2.VideoCapture(str(rendered_path))
    assert cap.isOpened()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count > 60

    # Verify download endpoint streams the MP4 file (FR-5.3)
    dl_res = client.get(download_url)
    assert dl_res.status_code == 200
    assert dl_res.headers["content-type"] == "video/mp4"
    assert len(dl_res.content) > 1000


def test_multiple_firearms_and_frontend_serving(
    client: TestClient,
    sample_video_path: Path,
):
    """Verify serving the frontend UI, demo upload endpoint, and multiple firearm profiles."""
    # Check frontend root HTML
    index_res = client.get("/")
    assert index_res.status_code == 200
    assert "Shooting Practice Video Analyzer" in index_res.text

    # Trigger demo upload for Sig P320
    demo_res = client.post("/api/demo-upload", data={"gun_name": "Sig P320"})
    assert demo_res.status_code == 200
    demo_session = demo_res.json()["session"]
    assert demo_session["gun_name"] == "Sig P320"
    assert demo_session["mode"] == "training"

    # Confirm annotations for Sig P320
    ann_res = client.post(
        "/api/annotations",
        json={
            "video_id": demo_session["video_id"],
            "gun_name": "Sig P320",
            "beep_timestamp": 1.00,
            "shot_timestamps": [2.25, 2.70, 3.10, 3.55],
            "user_modified": True,
        },
    )
    assert ann_res.status_code == 200

    # List firearms and verify Sig P320 is stored
    list_res = client.get("/api/firearms")
    assert list_res.status_code == 200
    names = [f["gun_name"] for f in list_res.json()["firearms"]]
    assert "Sig P320" in names


def test_config_default_not_storing_videos(
    client: TestClient,
    temp_storage: StorageManager,
):
    """Verify config.json defaults to not storing videos and cleans up previous session files."""
    cfg_res = client.get("/api/config")
    assert cfg_res.status_code == 200
    cfg_data = cfg_res.json()
    assert cfg_data["store_videos"] is False
    assert cfg_data["store_uploaded_videos"] is False
    assert cfg_data["store_rendered_videos"] is False

    # Create first demo session
    r1 = client.post("/api/demo-upload", data={"gun_name": "Glock 19"})
    vid_1 = r1.json()["session"]["video_id"]
    assert (temp_storage.uploads_dir / f"{vid_1}.mp4").exists()

    # Create second demo session -> first session's uploaded video must be deleted automatically
    r2 = client.post("/api/demo-upload", data={"gun_name": "Glock 19"})
    vid_2 = r2.json()["session"]["video_id"]
    assert not (temp_storage.uploads_dir / f"{vid_1}.mp4").exists()
    assert (temp_storage.uploads_dir / f"{vid_2}.mp4").exists()

    # Calling /api/cleanup deletes remaining temporary videos
    cleanup_res = client.post("/api/cleanup")
    assert cleanup_res.status_code == 200
    assert not (temp_storage.uploads_dir / f"{vid_2}.mp4").exists()


