# MobiCount

Automated counting pipeline for active-mobility users (pedestrians, cyclists, scooters, vehicles...) from fixed-camera videos, using YOLO detection and tracking (Ultralytics).

Developed by **Bruno Dato** (co-authors: **Adrien Lammoglia and Guillaume Dufour**) as part of urban mobility research (Université de Toulouse UT / IRIT, ONERA, LAGAM), with runs on local workstations (Windows/Linux) and cluster.

---

## Table of contents

- [MobiCount](#mobicount)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Repository structure](#repository-structure)
  - [Installation](#installation)
  - [Preparing videos (FFmpeg)](#preparing-videos-ffmpeg)
  - [Configuring a campaign (`DATA.py`)](#configuring-a-campaign-datapy)
  - [Video naming convention](#video-naming-convention)
  - [Two modes: `config` and `count`](#two-modes-config-and-count)
  - [Running a job](#running-a-job)
    - [Windows (local workstation / Casu)](#windows-local-workstation--casu)
    - [Linux](#linux)
    - [Cluster - SLURM (issues with ffmpeg for video compression)](#cluster---slurm-issues-with-ffmpeg-for-video-compression)
  - [Output files](#output-files)
  - [Hardware monitoring](#hardware-monitoring)
  - [Dependencies](#dependencies)
  - [License](#license)

---

## Overview

For each video, `MobiCount2.py`:

1. opens the video file with OpenCV;
2. runs an `ultralytics.solutions.ObjectCounter` (YOLO + BoT-SORT tracker) frame by frame, against a user-defined counting region (line or polygon);
3. detects IN/OUT crossings per object class (`person`, `bicycle`, `car`, `motorcycle`, `bus`, ...);
4. timestamps each counting event, based on the start time parsed from the video's filename;
5. writes an annotated video, then compresses it to `.mp4` (H.264) via FFmpeg;
6. exports two CSV files: a detailed event log and per-class totals.

The script can chain several videos in one run (`runSeveralCounts`), each with its own counting region, YOLO model and start date, all defined in `DATA.py`.

## Repository structure

```
MobiCount/
├── MobiCount2.py               # Main pipeline (counting + calibration)
├── MobiCount.py                # Legacy / single-video version (kept for reference)
├── MobiCount.ipynb             # Exploration notebook
├── DATA.py                     # Videos + regions + models to process (active campaign)
├── DATA_save.py                 # Other config sets / saved campaigns
├── RunOnWindows.py              # Local launch on Windows ("bruno" workstation)
├── RunOnWindows_2.py            # Local launch on Windows ("chloe" workstation)
├── RunOnCasu.py                 # Launch on Windows workstation 
├── RunOnLinux.py                # Launch on Linux workstation 
├── RunOnOccidata.py             # Launch on cluster
├── RunConfigOnCasu.py           # Calibration mode (Windows)
├── RunConfigOnOccidata.py       # Calibration mode (Occidata)
├── RunConfigOnWindows_2.py      # Calibration mode (Windows 2)
├── LauncherWindows.py / LauncherCasu.py / LauncherLinux.py / LauncherConfigCasu.py
│                                # Launch the Run* scripts in the background (detached subprocess.Popen)
├── FFMPEG_RemoveAudioCanal.py   # Utility: strips the audio track from a batch of videos
├── FFMPEG_LowerFPS.py           # Utility: resamples the FPS of a batch of videos
├── FFMPEG_CompressVideo.py      # Utility: compresses a video (H.265)
├── Occidata/
│   ├── runGPU.sh                 # SLURM job for counting (GPU partition, RTX8000)
│   ├── runCPU.sh                 # SLURM job for counting (CPU partition)
│   └── configGPU.sh              # SLURM job for calibration (builds the venv + runs RunConfigOnOccidata.py)
├── Monitoring/
│   ├── hw_logger.py              # CPU/GPU logger (psutil + pynvml) run in the background
│   └── README.md                 # Dedicated hardware-monitoring doc
├── Config/                       # Output of "config" mode (region calibration)
├── Video/                        # Source videos (.mp4) go here
├── Results/                      # Output of "count" mode (CSVs, annotated videos, frames)
├── requirements.txt              # Dependencies (workstation with a display)
├── requirementsHeadless.txt      # Dependencies (server/cluster, no display)
└── test.py                       # Ad-hoc test scripts
```

`Video/`, `Results/` and `Config/` are git-ignored (see `.gitignore`): these are local data folders you create/populate yourself.

## Installation

- Install **Python 3.12** (3.13 also works): https://www.python.org/downloads/release/python-3120/
- Install **FFmpeg**:
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux (Occidata): `module load ffmpeg/8.0`
- Create the virtual environment and install dependencies, from the parent folder of `MobiCount/`:

```bat
:: Windows (CMD)
PYTHON_PATH\python.exe -m venv .venv
.venv\Scripts\activate.bat
pip install -r MobiCount\requirements.txt
```

```bash
# Linux / cluster
python3 -m venv .venv
source .venv/bin/activate
pip install -r MobiCount/requirementsHeadless.txt
```

On a headless machine (server, cluster), use `requirementsHeadless.txt`, which installs `opencv-python-headless` and `dgenerate-ultralytics-headless` instead of the display-dependent versions.

Then create the `Video/`, `Results/` and `Config/` folders at the project root if they don't already exist, and drop the source videos into `Video/`.

## Preparing videos (FFmpeg)

A few useful commands before processing (also automated by the `FFMPEG_*.py` scripts):

```bash
# Strip the audio track
ffmpeg -i input.mp4 -c:v copy -an output.mp4
for f in *.mp4; do ffmpeg -i "$f" -c:v copy -an "../no-audio_${f}"; done          # bash
for %f in (*.mp4) do ffmpeg -i "%f" -c:v copy -an "../no-audio_%f"                 :: CMD

# Filter FPS
ffmpeg -i input.mp4 -filter:v "fps=30" output.mp4

# Extract the first 30 seconds
ffmpeg -i input.mp4 -t 30 -c copy output.mp4

# Rename *.MP4 -> *.mp4
for f in *.MP4; do mv "$f" "${f%.MP4}.mp4"; done

# Resize to HD (1920px wide)
for %f in (*.mp4) do ffmpeg -i "%f" -vf scale=1920:-2 -c:v libx264 -crf 23 -preset fast -c:a copy "../%~nf_HD.mp4"
foreach ($f in Get-ChildItem *.mp4) { ffmpeg -i $f.FullName -vf "scale=1920:-2" -c:v libx264 -crf 23 -preset fast -c:a copy "..\$($f.BaseName)_HD.mp4" }   # PowerShell
```

`FFMPEG_RemoveAudioCanal.py` and `FFMPEG_LowerFPS.py` apply the first two operations to an entire folder of videos (input/output paths and the FFmpeg executable path need to be adjusted at the top of each file).

## Configuring a campaign (`DATA.py`)

`DATA.py` (or a variant such as `DATA_save.py`, to be copied/renamed to `DATA.py` to activate it) defines, for a batch of videos processed in sequence:

| Variable | Role |
|---|---|
| `VIDEO_LIST` | Names of the videos (without extension) to process, in `Video/` |
| `DATES_LIST` | Reference date per video (mostly unused: the actual start time is parsed from the filename) |
| `REGION_LIST` | Counting line or polygon per video, in pixel coordinates: `[(x1,y1),(x2,y2)]` for a line, 4 points for a rectangle |
| `MODELS_LIST` | YOLO weights file to use per video (e.g. `"yolo11l"`, without the `.pt` extension) |

Videos excluded from the current run are simply commented out (`#`) in `VIDEO_LIST`.

## Video naming convention

The video filename encodes the start date and time, used to timestamp counting events. Expected format (segments separated by `_`):

```
no-audio_<point>_<DDMMYYYY>_<HH>_<MM>_<camera_letter>_<initial_resolution>_<fps>fps.mp4
```

Example: `no-audio_7.8_13052026_11_41_F_4k_25fps.mp4` → counting point `7.8`, camera `F`, starting at `13/05/2026 11:41`.

The parsing logic (`MobiCount2.py`) reads segments 3 to 5 exactly (`date_str`, `hour`, `minute`): any file that doesn't follow this format will break the date parsing.

## Two modes: `config` and `count`

`MobiCount2.py` exposes two functions, called in a loop over `VIDEO_LIST` by `runSeveralConfigs` / `runSeveralCounts`:

- **`config(...)`** — calibration mode: runs the detector on the first frame only and saves the annotated image (region + detections) to `Config/`, to visually check the placement of the counting line/polygon before running a full job. Default confidence: `0.1`.
- **`count(...)`** — full processing mode: goes through the whole video, accumulates IN/OUT counts per class, and writes the annotated video plus the CSVs. Default confidence: `0.01` (more permissive, so as not to miss detections across the whole video).

In both modes, detected classes are filtered to the COCO classes relevant to the study:

```python
CLASSES = [0, 1, 2, 3, 5]  # person, bicycle, car, motorcycle, bus
```

The tracker used is `botsort.yaml` (IoU = 0.9), and the inference device (`cpu` / `cuda:0`) is hardcoded in `ObjectCounter(...)` — adjust it depending on the machine used.

## Running a job

Each environment (workstation, cluster) has its own `RunOn*.py` script: it sets the FFmpeg path, the project folder, creates the day's results folder and log file, then calls `MobiCount2.runSeveralCounts(...)` (or `RunConfigOn*.py` → `runSeveralConfigs(...)` for calibration) with the lists defined in `DATA.py`.

### Windows (local workstation / Casu)

```bat
python RunOnWindows.py         :: Windows workstation
python RunOnWindows_2.py       :: Windows workstation
python RunOnCasu.py            :: Linux workstation
python RunConfigOnCasu.py      :: calibration on Linux
```

The matching `Launcher*.py` scripts (`LauncherWindows.py`, `LauncherCasu.py`, `LauncherConfigCasu.py`) launch these in the background via `subprocess.Popen` (stdout/stderr redirected to `DEVNULL`), handy for a detached launch from a scheduled task.

### Linux

```bash
python3 RunOnLinux.py    # Linux workstation
```
via `LauncherLinux.py` for a background launch.

### Cluster - SLURM (issues with ffmpeg for video compression)

Jobs are submitted from `Occidata/`:

```bash
sbatch Occidata/configGPU.sh   # region calibration (builds the venv, installs requirementsHeadless.txt, runs RunConfigOnOccidata.py)
sbatch Occidata/runGPU.sh      # full counting, GPU partition (RTX8000)
sbatch Occidata/runCPU.sh      # full counting, CPU partition (24CPUNodes)
```

These scripts load the `ffmpeg/8.0` and `Python/3.12.2` modules, activate the project's `.venv` (`/projects/campmob/`), then run the matching Python script. `configGPU.sh` sets `QT_QPA_PLATFORM=offscreen` to run without a display. SLURM logs are written to `logs/run_log_%j.out` / `logs/run_error_%j.err` (create the `logs/` folder if missing), in addition to the Python application logs written to `Results/<date>/` or `Config/<date>/`.

## Output files

For each video processed in `count` mode, under `Results/<YYYYMMDD>/`:

- `<timestamp>__<video>_<model>_counts.csv` — total IN/OUT per class for the whole video;
- `<timestamp>__<video>_<model>_events.csv` — one event per counting change, with the real-world time, frame number, per-class IN/OUT detail and the tracking IDs involved;
- `<timestamp>__<video>_<model>_FirstFrame.jpg` — first annotated frame (quick sanity check);
- `<timestamp>__<video>_<model>.mp4` — annotated video, recompressed to H.264 (0.5× resolution, FPS halved, CRF 30); the intermediate `.avi` file is deleted after compression.

In `config` mode, under `Config/<YYYYMMDD>/`: only the annotated `_FirstFrame.jpg` image, to validate the placement of the counting region.

Each run also writes a detailed log file (`<timestamp>.log`), including GPU memory available before/after processing and the full stack trace on crash (`SESSION CRASHED`).

## Hardware monitoring

`Monitoring/hw_logger.py` continuously logs CPU usage (overall + per core, frequencies, temperatures, RAM/swap, system load) and NVIDIA GPU usage (utilization, VRAM, temperature, power draw, clock speeds, processes) to a log file — useful for tracking load during a long run on the cluster:

```bash
pip install psutil pynvml   # pynvml = NVIDIA; gputil as an alternative
python Monitoring/hw_logger.py                  # 5 s interval -> hw_stats.log
python Monitoring/hw_logger.py 2                # every 2 s
python Monitoring/hw_logger.py 10 monitor.log   # every 10 s, named file
```

See `Monitoring/README.md` for the full log format.

## Dependencies

| File | Use case |
|---|---|
| `requirements.txt` | Workstation with a display: `opencv-python-headless`, `python-ffmpeg`, `ultralytics`, `shapely>=2.0.0`, `lap>=0.5.12`, `psutil`, `pynvml` |
| `requirementsHeadless.txt` | Headless server/cluster: same, but `dgenerate-ultralytics-headless` instead of `ultralytics` |

YOLO weights (`*.pt`, e.g. `yolo11l.pt`) are not versioned (see `.gitignore`) and must be present in the working directory. They are downloaded during first executions.

## License

Distributed under the **GNU AGPLv3** license (see `LICENSE`).
