# ZED Mapping Shareable

A shareable, trimmed project for the ZED SVO mapping GUI.

This repo contains the code needed to run the GUI, demo mode, and the underwater enhancement backend pipeline. Generated maps, analysis outputs, virtual environments, and local SVO captures are ignored by git.

## Included

- `src/zed_mapping_svo_gui.py`
- `src/underwater/`
- `requirements.txt`
- Windows setup and run scripts

Optional local data can be placed under `data/svo/`. The app looks for `data/svo/sofa.svo` on startup and falls back to Demo mode when that file is missing.

## What This App Does

- Loads a ZED SVO file and builds a live point-cloud view
- Supports live ZED camera streaming when a camera is connected and `pyzed` is available
- Supports a demo mode when `pyzed` is unavailable
- Exposes underwater enhancement controls in the GUI
- Keeps underwater processing in backend modules under `src/underwater/`

## Requirements

Base Python dependencies:
- Python 3.12 or newer
- `numpy`
- `PySide6`
- `pyvista`
- `pyvistaqt`
- `opencv-python`

For real ZED SVO playback or live camera streaming:
- Stereolabs ZED SDK installed
- ZED Python bindings available as `pyzed.sl`
- A supported Python version for your installed ZED SDK

Without the ZED SDK, the app still starts in Demo mode.

## Windows Setup

1. Install Python 3.12+.
2. Open a terminal in this folder.
3. Run:

```bat
setup_windows.bat
```

This creates `.venv` and installs the Python packages from `requirements.txt`.

## Run On Windows

```bat
run_gui.bat
```

If `.venv` exists, the launcher uses it automatically.

## ZED SDK Note

`pyzed` is not included in `requirements.txt` because it is installed through the official Stereolabs ZED SDK, not normal pip in most setups.

If a user has the ZED SDK installed correctly, the GUI can open an `.svo` or `.svo2` file from the source picker, or stream from a connected ZED camera through `Live ZED Camera` mode.

If not, the GUI falls back to Demo mode.

## GitHub / Git LFS

Local SVO captures are usually large and are ignored by default. If you intentionally want to publish one, track it with Git LFS and force-add the file:

Before pushing to GitHub:

```bat
git lfs install
git lfs track "*.svo" "*.svo2"
git add .gitattributes
git add -f data/svo/sofa.svo
```

If you do not want to use Git LFS, leave SVO files untracked and tell users to place their own local file at:
- `data/svo/sofa.svo`

## Project Layout

```text
zed_mapping_shareable/
  data/
    svo/
      .gitkeep
  src/
    __init__.py
    zed_mapping_svo_gui.py
    underwater/
      __init__.py
      config.py
      metrics.py
      ops.py
      pipeline.py
  .gitattributes
  .gitignore
  requirements.txt
  requirements-optional.txt
  run_gui.bat
  run_gui.ps1
  setup_windows.bat
  setup_windows.ps1
```

## Notes

- The GUI is responsible for controls, visualization, and diagnostics.
- The worker owns the underwater pipeline.
- If `cv2` is missing, CLAHE and denoise steps are skipped gracefully.
- Depth confidence integration remains a hook and only applies when real confidence data is available.
