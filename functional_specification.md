# Functional Specification: Shooting Practice Video Analyzer

## 1. Executive Summary
The Shooting Practice Video Analyzer is a web-based application designed for firearms practitioners to analyze their training sessions. By combining audio signature recognition, manual annotation, and automated detection, the system identifies timer beeps and gunshots within practice videos, computes precise split times, and generates annotated output videos with visual overlays.

---

## 2. Project Scope & Objectives
* **Unified Workflow:** Combine manual training/annotation and automated detection into a single intuitive user experience.
* **Audio Signature Learning:** Learn and refine acoustic profiles (timer beeps and specific gunshots) over time.
* **Precise Timing & Metrics:** Calculate time-to-first-shot (draw time) and individual split times between consecutive shots.
* **Visual Verification:** Provide a synchronized video/waveform player for editing markers and render processed videos with burnt-in performance overlays.

---

## 3. User Roles & Personas
* **Shooter / Practitioner:** Uploads videos, selects firearm models, reviews/corrects automated marker placements, and downloads annotated performance videos.

---

## 4. Detailed Functional Requirements

### 4.1. Module 1: Project Initialization & Upload
* **FR-1.1:** The system shall present a clean upload interface supporting standard video formats (e.g., `.mp4`, `.mov`, `.webm`).
* **FR-1.2:** The system shall require the user to select or input a firearm name (e.g., "Glock 19", "Sig P320") from a dropdown or text input.
* **FR-1.3:** Upon upload, the backend shall extract the audio track into a normalized mono PCM stream.

### 4.2. Module 2: Audio Profile & Signature Management
* **FR-2.1:** The system shall check local storage/database for an existing audio profile associated with the selected firearm name.
* **FR-2.2:** **Training Mode (Cold Start):** If no profile exists, the application enters manual annotation mode.
* **FR-2.3:** **Automated Mode:** If a profile exists, the backend runs automated peak detection and template matching using `librosa` to predict timer beep and gunshot timestamps.
* **FR-2.4:** **Profile Refinement:** Every time a user confirms or edits marker timestamps, the system updates the underlying acoustic feature vector (averaging spectral characteristics) to improve future automated detection accuracy.

### 4.3. Module 3: Interactive Waveform & Video Editor UI
* **FR-3.1:** The application shall render a synchronized HTML5 video player alongside an interactive audio waveform (using Wavesurfer.js).
* **FR-3.2:** The playback controls shall support Play, Pause, Stop, variable playback speed ($0.5x$ to $1.0x$), and timeline scrubbing.
* **FR-3.3:** The user shall be able to add, move, and delete markers for:
  * Timer Beep (Anchor point, $t = 0.00\text{s}$)
  * Individual Shots ($S_1, S_2, \dots, S_n$)
* **FR-3.4:** Keyboard shortcuts (e.g., `B` for Beep, `S` for Shot, `Delete` for removing selected marker) shall be supported for rapid annotation.

### 4.4. Module 4: Analytics & Split Calculation
* **FR-4.1:** The system shall calculate performance metrics based on confirmed timestamps:
  * **Draw / First Shot Time:** Time elapsed between Timer Beep and Shot 1.
  * **Split Times:** Time elapsed between consecutive shots ($S_i - S_{i-1}$).
  * **Total Stage Time:** Cumulative time from Beep to the final shot.
* **FR-4.2:** A summary table shall display all split times and metrics alongside the interactive timeline.

### 4.5. Module 5: Video Overlay Rendering
* **FR-5.1:** Upon clicking "Process Video", the backend (`opencv-python`) shall generate an annotated output video.
* **FR-5.2:** The video overlay shall display:
  * A clear visual indicator when the **Timer Beep** occurs ($t = 0.00\text{s}$).
  * A persistent or pop-up text overlay showing the **Shot Number**, **Split Time**, and **Total Elapsed Time** for 1–2 seconds following each shot event.
* **FR-5.3:** The system shall provide a download link for the final rendered video file.

---

## 5. Non-Functional Requirements
* **Performance:** Audio extraction and initial automated peak detection for a 10-second video clip should complete within 5 seconds on standard server hardware.
* **Usability:** The waveform interface must maintain smooth 60fps rendering during scrubbing and zooming.
* **Compatibility:** Modern web browsers (Chrome, Firefox, Safari, Edge) supporting HTML5 video and Web Audio API.

---

## 6. Data Models

### Firearm Audio Signature Schema
```json
{
  "gun_name": "string",
  "signature_version": "integer",
  "updated_at": "timestamp",
  "beep_profile": {
    "frequency_hz": "float",
    "duration_ms": "integer",
    "features": "array"
  },
  "shot_profile": {
    "decay_rate": "float",
    "features": "array"
  }
}
```

### Session Annotation Payload Schema
```json
{
  "video_id": "string",
  "gun_name": "string",
  "beep_timestamp": "float",
  "shot_timestamps": ["float"],
  "user_modified": "boolean"
}
```