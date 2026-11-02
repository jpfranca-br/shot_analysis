# Shooting Practice Video Analyzer (`shot_analysis`)

A web-based application designed for firearms practitioners to analyze training sessions. By combining **acoustic signature recognition** (`librosa`), **interactive waveform annotation** (`WaveSurfer.js`), and **automated transient detection**, the system identifies shot-timer beeps and gunshots within practice videos, computes precise draw and split times, and generates downloadable `.mp4` videos with burnt-in performance overlays (`opencv-python`).

---

## Key Features

- **Unified Training & Automated Workflow**
  - **Default Firearms Included:** Pre-configured with **CZ P-10C 9mm**, **T4 5.56x45mm**, and **CBC 7022 .22LR**.
  - **Firearm & Signature Model Manager:** Built-in UI manager (`⚙ Manage Firearms & Signatures`) to add new firearms, rename or edit acoustic parameters (Beep Frequency Hz, Duration ms, Shot Decay Rate dB/s), reset a learned signature back to Cold Start, remove firearms, or restore defaults.
  - **Cold Start (Training Mode):** When a new firearm model is used for the first time, manually annotate the timer beep and gunshots on the interactive waveform to train its acoustic profile.
  - **Automated Detection Mode (Dual-Regime + Reverb-Subtracted HF Flux):** Once a profile exists for a firearm, future uploads automatically detect the timer beep ($t = 0.00\text{s}$) and gunshot timestamps using:
    - **Reverb-Subtracted High-Frequency Spectral Flux ($1.8\text{–}10.5\text{ kHz}$):** Causal leaky-integrator reverberation floor subtraction and positive log-RMS attack tracking to unmask rapid follow-up shots riding on heavy indoor range reverberation tails.
    - **Dual-Regime Acoustic Signatures:** Separately models and matches **Isolated / First Shots** ($\Delta t > 0.85\text{s}$) and **Rapid Follow-up Shots** ($\Delta t \le 0.85\text{s}$, with pre-shot reverberation spectrum subtraction).
  - **Continuous Profile Refinement:** Every time you confirm or adjust markers, the firearm's acoustic signature vector (`signatures.json`) is refined and versioned (`v1`, `v2`, ...).
- **Synchronized Video & 60fps Waveform Editor**
  - HTML5 video player locked in sync with an interactive **WaveSurfer.js v7** waveform and timeline ruler.
  - Variable playback speed (`0.5x`, `0.75x`, `1.0x`), timeline scrubbing, and waveform zoom slider.
  - Drag-and-drop markers and rapid keyboard shortcuts:
    - <kbd>B</kbd> — Set Timer Beep anchor at playhead ($t = 0.00\text{s}$)
    - <kbd>S</kbd> — Add Gunshot marker at playhead ($S_1, S_2, \dots, S_n$)
    - <kbd>Delete</kbd> / <kbd>Backspace</kbd> — Remove selected marker
    - <kbd>Space</kbd> — Play / Pause
- **Live Split & Stage Analytics**
  - **Draw / First Shot Time:** Elapsed time from Timer Beep to Shot 1 ($S_1 - t_{\text{beep}}$).
  - **Split Times:** Elapsed time between consecutive shots ($S_i - S_{i-1}$).
  - **Total Stage Time:** Cumulative time from Timer Beep to the final shot ($S_n - t_{\text{beep}}$).
- **Burnt-In OpenCV Video Overlay Rendering**
  - Renders a downloadable `.mp4` video with original audio preserved and high-contrast tactical HUD overlays showing the **Timer Beep indicator**, **Shot Number**, **Split Time**, **Total Elapsed Time**, and a cumulative **Shot Log**.
- **Built-In Sample Drill Generator**
  - Includes a one-click **Load Sample Drill Video** button that synthesizes a range practice video with a 3000 Hz timer beep and 4 shots so you can test the entire workflow immediately.

---

## Project Structure

```text
shot_analysis/
├── backend/
│   ├── __init__.py
│   ├── main.py             # FastAPI application & REST endpoints
│   ├── config.py           # Storage configuration loader (config.json & env vars)
│   ├── models.py           # Pydantic data schemas (FirearmAudioSignature, SessionAnnotationPayload)
│   ├── storage.py          # Thread-safe local JSON store for signatures & sessions
│   ├── audio_engine.py     # Normalized mono PCM extraction, librosa feature learning & peak detection
│   ├── analytics.py        # Draw time, split time, and stage time computation
│   ├── video_engine.py     # OpenCV burnt-in HUD overlay renderer & VFR-synced H.264/AAC remuxer
│   └── demo_generator.py   # Synthetic shooting practice video generator for instant testing
├── frontend/
│   ├── index.html          # Single-page application layout
│   ├── styles.css          # Tactical dark-mode stylesheet
│   └── app.js              # WaveSurfer.js v7 & video synchronization controller
├── tests/
│   └── test_app.py         # End-to-end automated pytest suite
├── functional_specification.md
├── config.json             # Storage configuration (default: do not store videos)
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Configuration (`config.json`)

By default, **uploaded and rendered videos are not permanently stored on the server** (`"store_videos": false`).
- Temporary media files for the active session are automatically deleted when a new video is uploaded, when the server starts or stops, or when `/api/cleanup` is called.
- Learned firearm acoustic signatures (`data/signatures.json`) are always preserved.

You can customize this behavior in [`config.json`](./config.json):

```json
{
  "store_videos": false,
  "store_uploaded_videos": false,
  "store_rendered_videos": false
}
```

- Set `"store_videos": true` to keep both uploaded and rendered videos on the server (`data/uploads/` and `data/outputs/`).
- Or toggle `"store_uploaded_videos"` and `"store_rendered_videos"` individually.
- You can also override this at runtime with the environment variable `ANALYZER_STORE_VIDEOS=true`.

---

## Prerequisites

- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Git** (to clone the repository)
- **Modern Web Browser** (Chrome, Edge, Firefox, or Safari)

> **Note:** You do **not** need to install FFmpeg separately. The `imageio-ffmpeg` package listed in `requirements.txt` automatically bundles a self-contained FFmpeg binary for audio extraction and video remuxing.

---

## How to Download & Install

### 1. Clone the Repository

Open your terminal (PowerShell, Command Prompt, macOS Terminal, or Linux shell) and run:

```bash
git clone https://github.com/jpfranca-br/shot_analysis.git
cd shot_analysis
```

*(Alternatively, click **Code → Download ZIP** on GitHub, extract the archive, and open a terminal inside the extracted folder.)*

### 2. Create & Activate a Virtual Environment

> **Tip:** Using `--clear` ensures any previous virtual environment files (e.g., from a different Python version) are cleanly replaced.

**On Windows (PowerShell):**
```powershell
python -m venv --clear .venv
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv --clear .venv
.venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv --clear .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade `pip` and install the required packages from `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Running the Application

Start the FastAPI server with `uvicorn`:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Once the server starts, open your browser and navigate to:

```text
http://127.0.0.1:8000
```

---

## Quick Start Usage Guide

1. **Select or Manage a Firearm Model:**
   - Choose a preset (`CZ P-10C 9mm`, `T4 5.56x45mm`, `CBC 7022 .22LR`), choose `+ Custom Firearm Name...`, or click **⚙ Manage Firearms & Signatures** to add, rename, edit, reset, or remove firearm models.
2. **Upload a Video (or Load Sample Drill):**
   - Drag and drop a `.mp4`, `.mov`, or `.webm` shooting video and click **Analyze Video & Extract PCM Audio**, **OR** click **Load Sample Drill Video** to generate a ready-to-use practice video.
3. **Review or Annotate Markers:**
   - **Cold Start:** Scrub or play the video (`0.5x`–`1.0x`), press <kbd>B</kbd> (or click **+ Set Beep**) at the timer beep, and press <kbd>S</kbd> (or click **+ Add Shot**) at each gunshot peak.
   - **Automated Mode:** Review the automatically placed markers and drag any marker on the waveform to fine-tune its timestamp.
4. **Refine Profile & Render Video:**
   - Click **Confirm Markers & Refine Audio Profile** to save/update the acoustic signature for that firearm.
   - Click **⚙ Process Video with Burnt-In Overlays** to generate the annotated MP4 video, preview it directly in the browser, or click **⬇ Download Annotated Video (.MP4)**.

---

## Running Automated Tests

To run the automated test suite covering upload validation, mono PCM extraction, acoustic profile learning, automated `librosa` detection, split calculation, and OpenCV video rendering:

```bash
pytest -v
```
