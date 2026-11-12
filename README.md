# Batch FFT GUI

Small Tkinter GUI to process cryo-electron microscopy images in batch mode (FFT Gaussian low-pass filtering), with single-file preview tools.

<img width="761" height="499" alt="Interface_GH" src="https://github.com/user-attachments/assets/de2dea3c-cb2b-4800-b253-5e05c3b79cc9" />

## Features

- Open a folder and process all supported images in batch.
- Open a single file for interactive preview.
- Top-right logo support (`Logo_TEM2C.png` in project folder).
- Read pixel size from file headers (TIFF/MRC when available).
- Bin images by **1**, **2**, **4**, or **8** (bilinear resize).
- Auto-contrast to 8-bit before FFT filtering.
- Gaussian low-pass filter in Fourier space with configurable cut-off (Å).
- Save outputs with cut-off in filename: `name_cutoff.ext`.
- Save formats: **TIFF** (default), **PNG**, **JPEG**.
- Optional **16-bit output** (TIFF only).
- TIFF output preserves metadata when possible and updates:
  - image size
  - bit depth
  - pixel size / resolution tags
- Interactive preview window:
  - cut-off entry + apply
  - bin selector in preview
  - Image/FFT display toggle
  - axes visibility toggle (**Show axes** / **Hide axes**), default hidden
  - FFT colormap toggle (**Magma** / **Grey**)
  - contrast sliders + auto contrast
  - ROI-based auto-contrast (on image or FFT) with **Select ROI** button or right-click drag
  - histogram with gray-level axis (top-right panel)
  - zoom in/out, **Reset view**, pan-by-drag
  - mouse wheel zoom
  - line measurement tool with distance display
  - rotational average button to show a single split FFT frame (left: original spectrum, right: rotational average)
  - physical axes overlays (toggleable): image axes anchored to image bounds with origin at bottom-left (nm), FFT axes anchored to FFT bounds with 0,0 at FFT center (signed periodicity Å)

## Requirements

- Python 3.9+ (3.10+ recommended)
- Packages:
  - `numpy`
  - `Pillow`
  - `mrcfile` (needed only for `.mrc/.mrcs`)
- Tkinter (usually included with standard Python installers)

## Installation

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pillow mrcfile
```

> On Windows (PowerShell), activation is:
>
> ```powershell
> .venv\Scripts\Activate.ps1
> ```

## Run

```bash
python3 batch_fft.py
```

## How to Use

## 1) Main window

- **Open folder**: choose an input directory for batch processing.
- **Open file**: open one image in preview mode.
- **Cut-off (Å)**: FFT Gaussian filter cut-off.
- **Bin**: choose `1`, `2`, `4`, or `8` (default `2`).
- **Binned pixel size**: automatically updated from header pixel size × bin factor.
- **Save format**: TIFF / PNG / JPEG.
- **Bit depth**:
  - `8-bit` for all formats
  - `16-bit` for TIFF only
  - selecting PNG/JPEG forces 8-bit
- **Filter**: process all files in the selected folder.

Batch outputs are written into a new timestamped subfolder inside the input folder (e.g. `20261111_103000`).

## 2) Single-file preview window

- Adjust **Cut-off** and click **Apply** (or press Enter) to re-filter.
- Click **FFT** to switch between filtered image and FFT display (button toggles FFT/Image).
- Click **Show axes** to display axes, click **Hide axes** to hide them (default is hidden).
- Click **Magma/Grey** to toggle FFT rendering between magma and gray levels.
- Tune **Contrast min/max** or click **Auto**.
- Use **Select ROI** to draw a rectangle and apply ROI auto-contrast to the full current view.
- Right-click drag also performs ROI auto-contrast.
- Use **Zoom + / Zoom - / Reset view**.
- Click **Measure** then draw a line to measure distance:
  - Image mode: distance in nm
  - FFT mode: frequency spacing in 1/Å and periodicity estimate in Å
- Click **Rotational average** in FFT mode to switch to a single split FFT frame:
  - left half: original FFT spectrum
  - right half: rotationally averaged FFT image
- Left-click drag to pan.
- Use mouse wheel to zoom around cursor.
- Click **Save** to export the currently displayed image.

### Preview workflow example

1. Click **Open file** and select one image.
2. Set **Cut-off (Å)** (for example `20`) and click **Apply**.
3. Click **Auto** for a first contrast pass.
4. Click **Select ROI**, draw a rectangle on a meaningful area, and release mouse to update contrast from that ROI.
5. Click **FFT** to inspect frequency content, then (optionally) use **Select ROI** again in FFT mode to refine FFT contrast.
6. Use **Zoom + / Zoom - / Reset view**, mouse wheel, and drag-to-pan for detailed inspection.
7. Click **Save** to export the currently displayed result as `originalname_cutoff.ext`.

## Supported Input Formats

- `.mrc`, `.mrcs`
- `.tif`, `.tiff`
- `.png`, `.jpg`, `.jpeg`, `.bmp`

## Output Naming

Output files are saved as:

`original_name_<cutoff>.<ext>`

Examples:

- `sample_20.tif`
- `cell_10.jpg`

## Notes

- If pixel size is missing from header, the app falls back to `1.0 Å`.
- For physical consistency, FFT filtering uses the **binned pixel size**.
- FFT preview uses a display-only stabilization pipeline (mean-centering + edge apodization/Tukey window + centered square padding) to reduce vertical/horizontal border artifacts.
- FFT axes are drawn on the bottom/left edges of the displayed FFT and labeled as **signed periodicities (Å)** with 0 at FFT center.
- Image axes are drawn on the bottom/left edges of the displayed image with origin at **bottom-left (0,0)** in nm.

## Troubleshooting

- **`ModuleNotFoundError`**: install missing packages with `pip install numpy pillow mrcfile`.
- **No `.mrc` support**: verify `mrcfile` is installed in the same Python environment.
- **Tkinter import issues**:
  - Linux may require system package install (example Ubuntu: `sudo apt install python3-tk`).
- **Logo not shown**:
  - ensure `Logo_TEM2C.png` is in the same folder as `batch_fft.py`.
