# Shooting Practice Video Analyzer (`shot_analysis`)

A web-based application designed for firearms practitioners to analyze training sessions. By combining **acoustic signature recognition** (`librosa`), **interactive waveform annotation** (`WaveSurfer.js`), and **automated transient detection**, the system identifies shot-timer beeps and gunshots within practice videos, computes precise draw and split times, and generates downloadable `.mp4` videos with burnt-in performance overlays (`opencv-python`).

---

## Key Features

- **Unified Training & Automated Workflow**
  - **Cold Start (Training Mode):** When a new firearm model is used for the first time, manually annotate the timer beep and gunshots on the interactive waveform to train its acoustic profile.
  - **Automated Detection Mode:** Once a profile exists for a firearm (e.g., *Glock 19*, *Sig P320*, *CZ Shadow 2*), future uploads automatically detect the timer beep ($t = 0.00\text{s}$) and gunshot timestamps using STFT narrowband tonal matching, onset strength envelopes, and MFCC cosine similarity.
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
│   ├── models.py           # Pydantic data schemas (FirearmAudioSignature, SessionAnnotationPayload)
│   ├── storage.py          # Thread-safe local JSON store for signatures & sessions
│   ├── audio_engine.py     # Normalized mono PCM extraction, librosa feature learning & peak detection
│   ├── analytics.py        # Draw time, split time, and stage time computation
│   ├── video_engine.py     # OpenCV burnt-in HUD overlay renderer & H.264/AAC remuxer
│   └── demo_generator.py   # Synthetic shooting practice video generator for instant testing
├── frontend/
│   ├── index.html          # Single-page application layout
│   ├── styles.css          # Tactical dark-mode stylesheet
│   └── app.js              # WaveSurfer.js v7 & video synchronization controller
├── tests/
│   └── test_app.py         # End-to-end automated pytest suite
├── functional_specification.md
├── requirements.txt
├── pytest.ini
└── README.md
```

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
pip install -r requirements.txt
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

1. **Select or Enter a Firearm Model:**
   - Choose a preset (e.g., `Glock 19`, `Sig P320`, `CZ Shadow 2`) or choose `+ Custom Firearm Name...` to type any firearm model.
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
