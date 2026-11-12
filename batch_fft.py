"""Batch_ftt.py

GUI to process super-resolution cryo-EM images and save them in batch mode in different sizes and formats.

Individual files can be examined to choose the right parameters before batch processing.

Contact: denis.chretien@cnrs.fr

"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, TiffImagePlugin

try:
    import mrcfile  # type: ignore
except ImportError:
    mrcfile = None


IMAGE_EXTENSIONS = {".mrc", ".mrcs", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIFF_STRUCTURAL_TAGS = {
    256,  # ImageWidth
    257,  # ImageLength
    258,  # BitsPerSample
    259,  # Compression
    262,  # PhotometricInterpretation
    273,  # StripOffsets
    277,  # SamplesPerPixel
    278,  # RowsPerStrip
    279,  # StripByteCounts
    282,  # XResolution
    283,  # YResolution
    284,  # PlanarConfiguration
    296,  # ResolutionUnit
    317,  # Predictor
    322,  # TileWidth
    323,  # TileLength
    324,  # TileOffsets
    325,  # TileByteCounts
    339,  # SampleFormat
    530,  # YCbCrSubSampling
    531,  # YCbCrPositioning
}


def _build_magma_lut() -> np.ndarray:
    anchors = [
        (0, (0, 0, 4)),
        (64, (73, 15, 109)),
        (128, (182, 54, 121)),
        (192, (251, 136, 97)),
        (255, (252, 253, 191)),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for channel in range(3):
        x = [a[0] for a in anchors]
        y = [a[1][channel] for a in anchors]
        lut[:, channel] = np.interp(np.arange(256), x, y).astype(np.uint8)
    return lut


MAGMA_LUT = _build_magma_lut()


def list_image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _ratio_to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2 and value[1] != 0:
        return float(value[0]) / float(value[1])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_pixel_size_from_pil(image: Image.Image) -> Optional[float]:
    if not hasattr(image, "tag_v2"):
        return None
    tags = image.tag_v2
    x_resolution = _ratio_to_float(tags.get(282))  # pixels per unit
    unit_code = tags.get(296)  # 2 = inch, 3 = cm
    if not x_resolution or x_resolution <= 0:
        return None
    if unit_code == 2:
        unit_length_angstrom = 2.54e8  # 1 inch in Å
    elif unit_code == 3:
        unit_length_angstrom = 1.0e8  # 1 cm in Å
    else:
        return None
    return unit_length_angstrom / x_resolution


def build_tiffinfo(
    source_tags: dict[int, Any],
    width: int,
    height: int,
    pixel_size_angstrom: float,
    bits_per_sample: int,
) -> TiffImagePlugin.ImageFileDirectory_v2:
    tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
    for tag, value in source_tags.items():
        if tag in TIFF_STRUCTURAL_TAGS:
            continue
        try:
            tiffinfo[tag] = value
        except Exception:
            continue

    safe_pixel_size = max(pixel_size_angstrom, 1e-9)
    resolution_unit = int(source_tags.get(296, 3)) if source_tags.get(296, 3) in {2, 3} else 3
    unit_length_angstrom = 2.54e8 if resolution_unit == 2 else 1.0e8
    pixels_per_unit = unit_length_angstrom / safe_pixel_size

    tiffinfo[256] = width
    tiffinfo[257] = height
    tiffinfo[258] = bits_per_sample
    tiffinfo[282] = float(pixels_per_unit)
    tiffinfo[283] = float(pixels_per_unit)
    tiffinfo[296] = resolution_unit
    tiffinfo[339] = 1  # unsigned integer
    return tiffinfo


def read_image_and_pixel_size(path: Path) -> tuple[np.ndarray, Optional[float], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".mrc", ".mrcs"}:
        if mrcfile is None:
            raise RuntimeError("mrcfile is required for .mrc/.mrcs files. Install with: pip install mrcfile")
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.asarray(mrc.data, dtype=np.float32)
            pixel_size = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else None
        return data, pixel_size, {}

    with Image.open(path) as image:
        pixel_size = extract_pixel_size_from_pil(image)
        data = np.asarray(image, dtype=np.float32)
        metadata: dict[str, Any] = {}
        if image.format == "TIFF" and hasattr(image, "tag_v2"):
            metadata["tiff_tags"] = {int(tag): image.tag_v2.get(tag) for tag in image.tag_v2.keys()}
    return data, pixel_size, metadata


def to_grayscale(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data
    if data.ndim == 3:
        return np.mean(data, axis=2, dtype=np.float32)
    raise ValueError(f"Unsupported image dimensions: {data.shape}")


def bin_image_bilinear(data: np.ndarray, bin_factor: int) -> np.ndarray:
    if bin_factor not in {1, 2, 4, 8}:
        raise ValueError("Bin factor must be 1, 2, 4 or 8.")
    if bin_factor == 1:
        return data.astype(np.float32, copy=True)
    h, w = data.shape
    target_w = max(1, w // bin_factor)
    target_h = max(1, h // bin_factor)
    resized = Image.fromarray(data.astype(np.float32), mode="F").resize(
        (target_w, target_h),
        resample=Image.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def auto_contrast_to_uint8(data: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    lo = float(np.percentile(data, low_percentile))
    hi = float(np.percentile(data, high_percentile))
    if hi <= lo:
        return np.zeros_like(data, dtype=np.uint8)
    norm = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def cutoff_label(cutoff_angstrom: float) -> str:
    if float(cutoff_angstrom).is_integer():
        return str(int(cutoff_angstrom))
    return f"{cutoff_angstrom:.3f}".rstrip("0").rstrip(".")


def make_output_folder(base_folder: Path) -> Path:
    folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = base_folder / folder_name
    suffix = 1
    while output_folder.exists():
        output_folder = base_folder / f"{folder_name}_{suffix}"
        suffix += 1
    output_folder.mkdir(parents=True, exist_ok=False)
    return output_folder


def gaussian_lowpass_fft(data: np.ndarray, pixel_size_angstrom: float, cutoff_angstrom: float) -> np.ndarray:
    if pixel_size_angstrom <= 0:
        pixel_size_angstrom = 1.0
    cutoff_angstrom = max(cutoff_angstrom, 1e-6)

    rows, cols = data.shape
    fy = np.fft.fftfreq(rows, d=pixel_size_angstrom)
    fx = np.fft.fftfreq(cols, d=pixel_size_angstrom)
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    frequency_squared = fx_grid**2 + fy_grid**2

    sigma_frequency = 1.0 / cutoff_angstrom
    gaussian_filter = np.exp(-frequency_squared / (2.0 * sigma_frequency**2))

    frequency_data = np.fft.fft2(data.astype(np.float32))
    filtered = np.fft.ifft2(frequency_data * gaussian_filter).real
    return filtered.astype(np.float32)


def _tukey_window_1d(length: int, alpha: float = 0.2) -> np.ndarray:
    if length <= 1:
        return np.ones((max(1, length),), dtype=np.float32)
    if alpha <= 0.0:
        return np.ones((length,), dtype=np.float32)
    if alpha >= 1.0:
        n = np.arange(length, dtype=np.float32)
        return (0.5 * (1.0 - np.cos((2.0 * np.pi * n) / (length - 1)))).astype(np.float32)

    window = np.ones((length,), dtype=np.float32)
    edge = int(alpha * (length - 1) / 2.0)
    if edge <= 0:
        return window
    n = np.arange(edge + 1, dtype=np.float32)
    taper = 0.5 * (1.0 + np.cos(np.pi * ((2.0 * n / (alpha * (length - 1))) - 1.0)))
    window[: edge + 1] = taper
    window[-(edge + 1) :] = taper[::-1]
    return window


def square_fft_log_magnitude(data: np.ndarray) -> np.ndarray:
    centered = data.astype(np.float32, copy=False) - float(np.mean(data))
    rows, cols = centered.shape
    wy = _tukey_window_1d(rows, alpha=0.2)
    wx = _tukey_window_1d(cols, alpha=0.2)
    apodized = centered * np.outer(wy, wx)
    size = max(rows, cols)
    padded = np.zeros((size, size), dtype=np.float32)
    y0 = (size - rows) // 2
    x0 = (size - cols) // 2
    padded[y0 : y0 + rows, x0 : x0 + cols] = apodized
    fft_complex = np.fft.fftshift(np.fft.fft2(padded))
    return np.log1p(np.abs(fft_complex)).astype(np.float32)


def save_processed_image(
    output_path: Path,
    filtered_data: np.ndarray,
    save_format: str,
    bit_depth: str,
    metadata: dict[str, Any],
    pixel_size_angstrom: float,
) -> None:
    if bit_depth == "16-bit":
        output_16bit = (np.clip(filtered_data, 0, 255) * (65535.0 / 255.0)).astype(np.uint16)
        source_tags = metadata.get("tiff_tags", {})
        tiffinfo = build_tiffinfo(
            source_tags=source_tags,
            width=output_16bit.shape[1],
            height=output_16bit.shape[0],
            pixel_size_angstrom=pixel_size_angstrom,
            bits_per_sample=16,
        )
        Image.fromarray(output_16bit, mode="I;16").save(output_path, format="TIFF", tiffinfo=tiffinfo)
        return

    output_8bit = np.clip(filtered_data, 0, 255).astype(np.uint8)
    if save_format == "TIFF":
        source_tags = metadata.get("tiff_tags", {})
        tiffinfo = build_tiffinfo(
            source_tags=source_tags,
            width=output_8bit.shape[1],
            height=output_8bit.shape[0],
            pixel_size_angstrom=pixel_size_angstrom,
            bits_per_sample=8,
        )
        Image.fromarray(output_8bit, mode="L").save(output_path, format="TIFF", tiffinfo=tiffinfo)
    else:
        Image.fromarray(output_8bit, mode="L").save(output_path, format=save_format)


class SingleFilePreviewWindow(tk.Toplevel):
    def __init__(
        self,
        parent: "BatchFFTFilterGUI",
        source_path: Path,
        source_gray: np.ndarray,
        metadata: dict[str, Any],
        cutoff_angstrom: float,
        pixel_size_angstrom: float,
        bin_factor: int,
    ) -> None:
        super().__init__(parent)
        self.parent = parent
        self.source_path = source_path
        self.source_gray = source_gray.astype(np.float32)
        self.metadata = metadata
        self.cutoff_var = tk.StringVar(value=f"{cutoff_angstrom:.3f}".rstrip("0").rstrip("."))
        self.base_pixel_size_angstrom = pixel_size_angstrom
        self.pixel_size_var = tk.StringVar(value=f"{pixel_size_angstrom * float(bin_factor):.4f}")
        self.bin_factor_var = tk.StringVar(value=str(bin_factor))
        self.current_display_8bit: np.ndarray = np.zeros((1, 1), dtype=np.uint8)
        self.current_filtered_data: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.current_fft_data: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.current_fft_rotavg_data: np.ndarray = np.zeros((1, 1), dtype=np.float32)
        self.current_binned_pixel_size = pixel_size_angstrom * float(bin_factor)
        self.display_mode = "image"
        self.fft_pair_mode = False
        self.show_axes = False
        self.fft_colormap = "magma"
        self.tk_preview_image: Optional[ImageTk.PhotoImage] = None
        self.zoom_scale = 1.0
        self.image_offset_x = 0.0
        self.image_offset_y = 0.0
        self._pan_start_x = 0
        self._pan_start_y = 0
        self.roi_mode_enabled = False
        self.roi_start_x = 0
        self.roi_start_y = 0
        self.roi_rect_id: Optional[int] = None
        self.roi_button_default_bg = ""
        self.measure_mode_enabled = False
        self.measure_line_id: Optional[int] = None
        self.measure_start_x = 0
        self.measure_start_y = 0
        self.measure_button_default_bg = ""
        self.measure_var = tk.StringVar(value="Distance: N/A")
        self.preview_save_folder: Optional[Path] = None

        self.title(f"Preview: {source_path.name}")
        self.geometry("1160x760")

        self.contrast_min_var = tk.DoubleVar(value=0.0)
        self.contrast_max_var = tk.DoubleVar(value=255.0)

        self._build_ui()
        self.bin_factor_var.trace_add("write", self._on_preview_bin_changed)
        self.apply_cutoff_and_refresh(auto_contrast=True)

    def _on_preview_bin_changed(self, *_: object) -> None:
        try:
            bin_factor = int(self.bin_factor_var.get())
            if bin_factor not in {1, 2, 4, 8}:
                return
        except ValueError:
            return
        self.pixel_size_var.set(f"{self.base_pixel_size_angstrom * float(bin_factor):.4f}")

    def _build_ui(self) -> None:
        control = tk.Frame(self, padx=10, pady=10)
        control.pack(fill="x")

        top_row = tk.Frame(control)
        top_row.pack(fill="x")
        tk.Label(top_row, text="Cut-off (Å)").pack(side="left")
        self.cutoff_entry = tk.Entry(top_row, textvariable=self.cutoff_var, width=4)
        self.cutoff_entry.pack(side="left", padx=(8, 8))
        self.cutoff_entry.bind("<Return>", lambda _: self.apply_cutoff_and_refresh(auto_contrast=False))
        tk.Label(top_row, text="Bin").pack(side="left", padx=(8, 0))
        tk.OptionMenu(top_row, self.bin_factor_var, "1", "2", "4", "8").pack(side="left", padx=(8, 8))
        tk.Label(top_row, text="Pixel size (Å/px)").pack(side="left", padx=(8, 0))
        self.preview_pixel_size_entry = tk.Entry(top_row, textvariable=self.pixel_size_var, width=7)
        self.preview_pixel_size_entry.pack(side="left", padx=(8, 8))
        self.preview_pixel_size_entry.bind("<Return>", lambda _: self.apply_cutoff_and_refresh(auto_contrast=False))
        tk.Button(top_row, text="Apply", command=lambda: self.apply_cutoff_and_refresh(auto_contrast=False), width=10).pack(
            side="left", padx=(0, 10)
        )
        tk.Button(top_row, text="Save", command=self.save_current_image, width=10).pack(side="left", padx=(6, 0))
        self.fft_button = tk.Button(top_row, text="FFT", command=self.toggle_fft_view, width=10)
        self.fft_button.pack(side="left", padx=(6, 0))
        self.axes_button = tk.Button(top_row, text="Show axes", command=self.toggle_axes_visibility, width=10)
        self.axes_button.pack(side="left", padx=(6, 0))
        self.fft_colormap_button = tk.Button(top_row, text="Magma", command=self.toggle_fft_colormap, width=10)
        self.fft_colormap_button.pack(side="left", padx=(6, 0))

        contrast_row = tk.Frame(control)
        contrast_row.pack(fill="x", pady=(8, 0))
        tk.Label(contrast_row, text="Contrast min").grid(row=0, column=0, sticky="w")
        self.contrast_min_scale = tk.Scale(
            contrast_row,
            variable=self.contrast_min_var,
            from_=0.0,
            to=255.0,
            orient="horizontal",
            resolution=0.1,
            command=lambda _: self.update_preview(),
            length=320,
        )
        self.contrast_min_scale.grid(row=0, column=1, columnspan=3, padx=(8, 20), sticky="ew")

        tk.Label(contrast_row, text="Contrast max").grid(row=0, column=4, sticky="w")
        self.contrast_max_scale = tk.Scale(
            contrast_row,
            variable=self.contrast_max_var,
            from_=0.0,
            to=255.0,
            orient="horizontal",
            resolution=0.1,
            command=lambda _: self.update_preview(),
            length=320,
        )
        self.contrast_max_scale.grid(row=0, column=5, columnspan=2, padx=(8, 20), sticky="ew")

        tk.Button(contrast_row, text="Auto", command=self.auto_contrast, width=10).grid(row=0, column=7, sticky="w")
        self.roi_button = tk.Button(contrast_row, text="Select ROI", command=self.toggle_roi_mode, width=12)
        self.roi_button.grid(row=0, column=8, padx=(6, 0), sticky="w")
        self.roi_button_default_bg = self.roi_button.cget("bg")
        contrast_row.grid_columnconfigure(1, weight=1)
        contrast_row.grid_columnconfigure(5, weight=1)

        content = tk.Frame(self, padx=10, pady=10)
        content.pack(fill="both", expand=True)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=0)
        content.grid_rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(content, bd=1, relief="solid", bg="black")
        self.image_canvas.grid(row=0, column=0, sticky="nsew")
        self.image_canvas.bind("<Configure>", lambda _: self.fit_to_canvas())
        self.image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.image_canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.image_canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.image_canvas.bind("<ButtonPress-3>", self.on_roi_press)
        self.image_canvas.bind("<B3-Motion>", self.on_roi_drag)
        self.image_canvas.bind("<ButtonRelease-3>", self.on_roi_release)
        self.image_canvas.bind("<ButtonPress-2>", self.on_roi_press)
        self.image_canvas.bind("<B2-Motion>", self.on_roi_drag)
        self.image_canvas.bind("<ButtonRelease-2>", self.on_roi_release)

        right_panel = tk.Frame(content, padx=10)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=2)

        hist_frame = tk.Frame(right_panel)
        hist_frame.grid(row=0, column=0, sticky="nsew")
        tk.Label(hist_frame, text="Histogram").pack(anchor="n")
        self.hist_canvas = tk.Canvas(hist_frame, width=140, height=120, bd=1, relief="solid", bg="white")
        self.hist_canvas.pack(fill="both", expand=True, pady=(6, 0))
        self.hist_canvas.bind("<Configure>", lambda _: self.draw_histogram(self.current_display_8bit))

        bottom_right = tk.Frame(right_panel)
        bottom_right.grid(row=1, column=0, sticky="nsew")
        self.mode_label = tk.Label(bottom_right, text="Mode: Image", anchor="n")
        self.mode_label.pack(anchor="n", pady=(6, 0))
        tk.Button(bottom_right, text="Zoom +", command=lambda: self.zoom_at_center(1.25), width=10).pack(pady=(12, 0))
        tk.Button(bottom_right, text="Zoom -", command=lambda: self.zoom_at_center(0.8), width=10).pack(pady=(6, 0))
        tk.Button(bottom_right, text="Reset view", command=self.fit_to_canvas, width=10).pack(pady=(6, 0))
        self.measure_button = tk.Button(bottom_right, text="Measure", command=self.toggle_measure_mode, width=10)
        self.measure_button.pack(pady=(8, 0))
        self.rotavg_button = tk.Button(bottom_right, text="Rotational average", command=self.toggle_rotational_average_view, width=14)
        self.rotavg_button.pack(pady=(8, 0))
        self.measure_button_default_bg = self.measure_button.cget("bg")
        tk.Label(bottom_right, textvariable=self.measure_var, justify="left", wraplength=150).pack(pady=(8, 0))

    def apply_cutoff_and_refresh(self, auto_contrast: bool) -> None:
        try:
            cutoff = float(self.cutoff_var.get())
            if cutoff <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid cut-off", "Cut-off must be a positive number (Å).")
            return
        try:
            binned_pixel_size = float(self.pixel_size_var.get())
            if binned_pixel_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid pixel size", "Pixel size must be a positive number (Å/px).")
            return
        try:
            bin_factor = int(self.bin_factor_var.get())
            if bin_factor not in {1, 2, 4, 8}:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid binning", "Bin factor must be 1, 2, 4 or 8.")
            return

        self.base_pixel_size_angstrom = binned_pixel_size / float(bin_factor)
        binned = bin_image_bilinear(self.source_gray, bin_factor)
        contrasted = auto_contrast_to_uint8(binned).astype(np.float32)
        self.current_binned_pixel_size = binned_pixel_size

        self.current_filtered_data = gaussian_lowpass_fft(contrasted, self.current_binned_pixel_size, cutoff)
        self.current_fft_data = square_fft_log_magnitude(self.current_filtered_data)
        self.current_fft_rotavg_data = self._build_rotational_average_image(self.current_fft_data)
        self.fft_pair_mode = False
        self.rotavg_button.configure(text="Rotational average")
        self._set_mode_label()
        if auto_contrast:
            self.auto_contrast()
        else:
            self._update_contrast_slider_range()
            self.update_preview()
        self.fit_to_canvas()

    def _get_active_data(self) -> np.ndarray:
        if self.display_mode == "fft":
            if self.fft_pair_mode:
                fft_mix = self.current_fft_data.copy()
                split_col = fft_mix.shape[1] // 2
                fft_mix[:, split_col:] = self.current_fft_rotavg_data[:, split_col:]
                return fft_mix.astype(np.float32)
            return self.current_fft_data
        return self.current_filtered_data

    def _set_mode_label(self) -> None:
        if self.display_mode == "fft" and self.fft_pair_mode:
            self.mode_label.configure(text="Mode: FFT + RotAvg")
        else:
            self.mode_label.configure(text="Mode: FFT" if self.display_mode == "fft" else "Mode: Image")

    def toggle_roi_mode(self) -> None:
        self.roi_mode_enabled = not self.roi_mode_enabled
        if self.roi_mode_enabled:
            if self.measure_mode_enabled:
                self.toggle_measure_mode()
            self.roi_button.configure(
                relief="sunken",
                bg="#ff7f7f",
                highlightthickness=2,
                highlightbackground="red",
                highlightcolor="red",
            )
            self.image_canvas.configure(cursor="crosshair")
        else:
            self.roi_button.configure(
                relief="raised",
                bg=self.roi_button_default_bg,
                highlightthickness=0,
                highlightbackground=self.roi_button_default_bg,
                highlightcolor=self.roi_button_default_bg,
            )
            self.image_canvas.configure(cursor="")
            if self.roi_rect_id is not None:
                self.image_canvas.delete(self.roi_rect_id)
                self.roi_rect_id = None

    def toggle_measure_mode(self) -> None:
        self.measure_mode_enabled = not self.measure_mode_enabled
        if self.measure_mode_enabled:
            if self.roi_mode_enabled:
                self.toggle_roi_mode()
            self.measure_button.configure(
                relief="sunken",
                bg="#ff7f7f",
                highlightthickness=2,
                highlightbackground="red",
                highlightcolor="red",
            )
            self.image_canvas.configure(cursor="crosshair")
        else:
            self.measure_button.configure(
                relief="raised",
                bg=self.measure_button_default_bg,
                highlightthickness=0,
                highlightbackground=self.measure_button_default_bg,
                highlightcolor=self.measure_button_default_bg,
            )
            self.image_canvas.configure(cursor="")
            if self.measure_line_id is not None:
                self.image_canvas.delete(self.measure_line_id)
                self.measure_line_id = None

    def toggle_fft_view(self) -> None:
        self.display_mode = "fft" if self.display_mode == "image" else "image"
        if self.display_mode == "image":
            self.fft_pair_mode = False
            self.rotavg_button.configure(text="Rotational average")
        self.fft_button.configure(text="Image" if self.display_mode == "fft" else "FFT")
        self._set_mode_label()
        self.auto_contrast()

    def toggle_axes_visibility(self) -> None:
        self.show_axes = not self.show_axes
        self.axes_button.configure(text="Hide axes" if self.show_axes else "Show axes")
        self.render_canvas_image()

    def toggle_fft_colormap(self) -> None:
        self.fft_colormap = "grey" if self.fft_colormap == "magma" else "magma"
        self.fft_colormap_button.configure(text="Grey" if self.fft_colormap == "grey" else "Magma")
        self.render_canvas_image()

    def _update_contrast_slider_range(self) -> None:
        active_data = self._get_active_data()
        data_min = float(np.min(active_data))
        data_max = float(np.max(active_data))
        if data_max <= data_min:
            data_max = data_min + 1.0
        for slider_var in (self.contrast_min_var, self.contrast_max_var):
            current_value = float(slider_var.get())
            slider_var.set(min(max(current_value, data_min), data_max))
        self.contrast_min_scale.configure(from_=data_min, to=data_max)
        self.contrast_max_scale.configure(from_=data_min, to=data_max)

    def auto_contrast(self) -> None:
        self._update_contrast_slider_range()
        active_data = self._get_active_data()
        low = float(np.percentile(active_data, 1.0))
        high = float(np.percentile(active_data, 99.0))
        if high <= low:
            high = low + 1.0
        self.contrast_min_var.set(low)
        self.contrast_max_var.set(high)
        self.update_preview()

    def update_preview(self) -> None:
        lo = float(self.contrast_min_var.get())
        hi = float(self.contrast_max_var.get())
        if hi <= lo:
            hi = lo + 1e-6
            self.contrast_max_var.set(hi)

        active_data = self._get_active_data()
        norm = np.clip((active_data - lo) / (hi - lo), 0.0, 1.0)
        display = (norm * 255.0).astype(np.uint8)
        self.current_display_8bit = display

        self.render_canvas_image()
        self.draw_histogram(display)

    def draw_histogram(self, display_8bit: np.ndarray) -> None:
        if display_8bit.size == 0:
            return
        hist, _ = np.histogram(display_8bit, bins=256, range=(0, 255))
        max_hist = int(hist.max()) if hist.size else 1
        if max_hist <= 0:
            max_hist = 1

        self.hist_canvas.delete("all")
        self.hist_canvas.update_idletasks()
        width = max(int(self.hist_canvas.winfo_width()), 140)
        height = max(int(self.hist_canvas.winfo_height()), 120)
        left_margin = 28
        right_margin = 8
        top_margin = 8
        bottom_margin = 26
        plot_width = max(10, width - left_margin - right_margin)
        plot_height = max(10, height - top_margin - bottom_margin)
        x_step = plot_width / 256.0

        for i, count in enumerate(hist):
            x0 = left_margin + i * x_step
            x1 = left_margin + (i + 1) * x_step
            y1 = top_margin + plot_height
            y0 = y1 - (count / max_hist) * plot_height
            self.hist_canvas.create_rectangle(x0, y0, x1, y1, fill="#505050", outline="#505050")

        axis_y = top_margin + plot_height
        self.hist_canvas.create_line(left_margin, axis_y, left_margin + plot_width, axis_y, fill="black")
        for tick in (0, 64, 128, 192, 255):
            x = left_margin + (tick / 255.0) * plot_width
            self.hist_canvas.create_line(x, axis_y, x, axis_y + 4, fill="black")
            self.hist_canvas.create_text(x, axis_y + 12, text=str(tick), font=("TkDefaultFont", 7))
        self.hist_canvas.create_text(left_margin + plot_width / 2.0, height - 7, text="Gray levels", font=("TkDefaultFont", 8))

    def render_canvas_image(self) -> None:
        if self.current_display_8bit.size == 0:
            return
        src_h, src_w = self.current_display_8bit.shape
        target_w = max(1, int(round(src_w * self.zoom_scale)))
        target_h = max(1, int(round(src_h * self.zoom_scale)))
        if self.display_mode == "fft":
            if self.fft_colormap == "magma":
                rgb = MAGMA_LUT[self.current_display_8bit]
                resized = Image.fromarray(rgb, mode="RGB").resize((target_w, target_h), Image.NEAREST)
            else:
                resized = Image.fromarray(self.current_display_8bit, mode="L").resize((target_w, target_h), Image.NEAREST)
        else:
            resized = Image.fromarray(self.current_display_8bit, mode="L").resize((target_w, target_h), Image.NEAREST)
        self.tk_preview_image = ImageTk.PhotoImage(resized)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(self.image_offset_x, self.image_offset_y, image=self.tk_preview_image, anchor="nw")
        self.draw_axes_overlay()

    def draw_axes_overlay(self) -> None:
        if not self.show_axes or self.current_display_8bit.size == 0:
            return
        src_h, src_w = self.current_display_8bit.shape
        canvas_w = max(1, int(self.image_canvas.winfo_width()))
        canvas_h = max(1, int(self.image_canvas.winfo_height()))

        pixel_size = max(self.current_binned_pixel_size, 1e-9)
        axis_color = "#2f80ff" if self.display_mode == "fft" else "#ff3b3b"
        tick_font = ("TkDefaultFont", 11, "bold")
        unit_font = ("TkDefaultFont", 12, "bold")
        if self.display_mode == "fft":
            fft_h, fft_w = self.current_fft_data.shape
            x0 = self.image_offset_x
            x1 = self.image_offset_x + fft_w * self.zoom_scale
            y1 = self.image_offset_y
            y0 = self.image_offset_y + fft_h * self.zoom_scale
            if x1 <= x0 or y0 <= y1:
                return
            self.image_canvas.create_line(x0, y0, x1, y0, fill=axis_color, width=2)
            self.image_canvas.create_line(x0, y0, x0, y1, fill=axis_color, width=2)

            x_ticks = 5
            y_ticks = 5
            for i in range(x_ticks):
                t = i / float(x_ticks - 1)
                px = x0 + t * (x1 - x0)
                pix_coord = t * fft_w
                center = fft_w / 2.0
                frequency = (pix_coord - center) / (fft_w * pixel_size)
                periodicity = 0.0 if abs(frequency) < 5e-5 else 1.0 / frequency
                self.image_canvas.create_line(px, y0, px, y0 + 5, fill=axis_color, width=1)
                self.image_canvas.create_text(px, y0 + 18, anchor="n", fill=axis_color, text=f"{periodicity:.1f}", font=tick_font)

            for i in range(y_ticks):
                t = i / float(y_ticks - 1)
                py = y0 - t * (y0 - y1)
                pix_coord = fft_h - t * fft_h
                center = fft_h / 2.0
                frequency = (center - pix_coord) / (fft_h * pixel_size)
                periodicity = 0.0 if abs(frequency) < 5e-5 else 1.0 / frequency
                self.image_canvas.create_line(x0 - 5, py, x0, py, fill=axis_color, width=1)
                self.image_canvas.create_text(x0 - 10, py, anchor="e", fill=axis_color, text=f"{periodicity:.1f}", font=tick_font)

            self.image_canvas.create_text((x0 + x1) / 2.0, y0 + 34, anchor="n", fill=axis_color, text="X (Å)", font=unit_font)
            self.image_canvas.create_text(
                x0 - 56,
                (y0 + y1) / 2.0,
                anchor="center",
                fill=axis_color,
                text="Y (Å)",
                font=unit_font,
                angle=90,
            )
            return

        x0 = self.image_offset_x
        x1 = self.image_offset_x + src_w * self.zoom_scale
        y1 = self.image_offset_y
        y0 = self.image_offset_y + src_h * self.zoom_scale
        if x1 <= x0 or y0 <= y1:
            return
        self.image_canvas.create_line(x0, y0, x1, y0, fill=axis_color, width=2)
        self.image_canvas.create_line(x0, y0, x0, y1, fill=axis_color, width=2)

        x_ticks = 5
        y_ticks = 5
        for i in range(x_ticks):
            t = i / float(x_ticks - 1)
            px = x0 + t * (x1 - x0)
            x_nm = t * src_w * pixel_size * 0.1
            label = f"{x_nm:.1f}"
            x_unit = "nm"
            self.image_canvas.create_line(px, y0, px, y0 + 5, fill=axis_color, width=1)
            self.image_canvas.create_text(px, y0 + 18, anchor="n", fill=axis_color, text=label, font=tick_font)

        for i in range(y_ticks):
            t = i / float(y_ticks - 1)
            py = y0 - t * (y0 - y1)
            y_nm = t * src_h * pixel_size * 0.1
            label = f"{y_nm:.1f}"
            y_unit = "nm"
            self.image_canvas.create_line(x0 - 5, py, x0, py, fill=axis_color, width=1)
            self.image_canvas.create_text(x0 - 10, py, anchor="e", fill=axis_color, text=label, font=tick_font)

        self.image_canvas.create_text((x0 + x1) / 2.0, y0 + 34, anchor="n", fill=axis_color, text=f"X ({x_unit})", font=unit_font)
        self.image_canvas.create_text(
            x0 - 56,
            (y0 + y1) / 2.0,
            anchor="center",
            fill=axis_color,
            text=f"Y ({y_unit})",
            font=unit_font,
            angle=90,
        )

    def fit_to_canvas(self) -> None:
        if self.current_display_8bit.size == 0:
            return
        canvas_w = max(1, int(self.image_canvas.winfo_width()))
        canvas_h = max(1, int(self.image_canvas.winfo_height()))
        src_h, src_w = self.current_display_8bit.shape
        fit_scale = min(canvas_w / float(src_w), canvas_h / float(src_h))
        self.zoom_scale = max(0.05, fit_scale)
        img_w = src_w * self.zoom_scale
        img_h = src_h * self.zoom_scale
        self.image_offset_x = (canvas_w - img_w) / 2.0
        self.image_offset_y = (canvas_h - img_h) / 2.0
        self.render_canvas_image()

    def zoom_at_center(self, factor: float) -> None:
        canvas_w = max(1, int(self.image_canvas.winfo_width()))
        canvas_h = max(1, int(self.image_canvas.winfo_height()))
        self.zoom_at_point(canvas_w / 2.0, canvas_h / 2.0, factor)

    def zoom_at_point(self, x: float, y: float, factor: float) -> None:
        if self.current_display_8bit.size == 0:
            return
        src_h, src_w = self.current_display_8bit.shape
        old_scale = self.zoom_scale
        new_scale = max(0.05, min(20.0, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-9:
            return
        source_x = (x - self.image_offset_x) / old_scale
        source_y = (y - self.image_offset_y) / old_scale
        self.zoom_scale = new_scale
        self.image_offset_x = x - source_x * new_scale
        self.image_offset_y = y - source_y * new_scale
        self.render_canvas_image()

    def on_canvas_press(self, event: tk.Event[tk.Misc]) -> None:
        if self.measure_mode_enabled:
            self.measure_start_x = int(event.x)
            self.measure_start_y = int(event.y)
            if self.measure_line_id is not None:
                self.image_canvas.delete(self.measure_line_id)
            self.measure_line_id = self.image_canvas.create_line(
                self.measure_start_x,
                self.measure_start_y,
                self.measure_start_x,
                self.measure_start_y,
                fill="yellow",
                width=2,
            )
            return
        if self.roi_mode_enabled:
            self.on_roi_press(event)
            return
        self._pan_start_x = int(event.x)
        self._pan_start_y = int(event.y)

    def on_canvas_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.measure_mode_enabled:
            if self.measure_line_id is not None:
                self.image_canvas.coords(
                    self.measure_line_id,
                    self.measure_start_x,
                    self.measure_start_y,
                    int(event.x),
                    int(event.y),
                )
            return
        if self.roi_mode_enabled:
            self.on_roi_drag(event)
            return
        dx = int(event.x) - self._pan_start_x
        dy = int(event.y) - self._pan_start_y
        self.image_offset_x += dx
        self.image_offset_y += dy
        self._pan_start_x = int(event.x)
        self._pan_start_y = int(event.y)
        self.render_canvas_image()

    def on_canvas_release(self, event: tk.Event[tk.Misc]) -> None:
        if self.measure_mode_enabled:
            x0, y0 = self.measure_start_x, self.measure_start_y
            x1, y1 = int(event.x), int(event.y)
            active_data = self._get_active_data()
            src_h, src_w = active_data.shape
            sx0 = (x0 - self.image_offset_x) / self.zoom_scale
            sy0 = (y0 - self.image_offset_y) / self.zoom_scale
            sx1 = (x1 - self.image_offset_x) / self.zoom_scale
            sy1 = (y1 - self.image_offset_y) / self.zoom_scale
            sx0 = float(np.clip(sx0, 0.0, max(0, src_w - 1)))
            sx1 = float(np.clip(sx1, 0.0, max(0, src_w - 1)))
            sy0 = float(np.clip(sy0, 0.0, max(0, src_h - 1)))
            sy1 = float(np.clip(sy1, 0.0, max(0, src_h - 1)))
            dx_pix = sx1 - sx0
            dy_pix = sy1 - sy0
            if self.display_mode == "fft":
                px = max(self.current_binned_pixel_size, 1e-9)
                fx0 = (sx0 - src_w / 2.0) / (src_w * px)
                fy0 = (sy0 - src_h / 2.0) / (src_h * px)
                fx1 = (sx1 - src_w / 2.0) / (src_w * px)
                fy1 = (sy1 - src_h / 2.0) / (src_h * px)
                df = float(np.hypot(fx1 - fx0, fy1 - fy0))
                if df < 1e-12:
                    self.measure_var.set("Distance: Δf≈0 (periodicity ∞)")
                else:
                    self.measure_var.set(f"Distance: Δf={df:.4f} 1/Å (d={1.0 / df:.1f} Å)")
            else:
                distance_nm = float(np.hypot(dx_pix, dy_pix) * self.current_binned_pixel_size * 0.1)
                self.measure_var.set(f"Distance: {distance_nm:.2f} nm")
            return
        if self.roi_mode_enabled:
            self.on_roi_release(event)

    def on_roi_press(self, event: tk.Event[tk.Misc]) -> None:
        self.roi_start_x = int(event.x)
        self.roi_start_y = int(event.y)
        if self.roi_rect_id is not None:
            self.image_canvas.delete(self.roi_rect_id)
        self.roi_rect_id = self.image_canvas.create_rectangle(
            self.roi_start_x,
            self.roi_start_y,
            self.roi_start_x,
            self.roi_start_y,
            outline="red",
            width=2,
        )

    def on_roi_drag(self, event: tk.Event[tk.Misc]) -> None:
        if self.roi_rect_id is not None:
            self.image_canvas.coords(self.roi_rect_id, self.roi_start_x, self.roi_start_y, int(event.x), int(event.y))

    def on_roi_release(self, event: tk.Event[tk.Misc]) -> None:
        x0, y0 = self.roi_start_x, self.roi_start_y
        x1, y1 = int(event.x), int(event.y)
        if self.roi_rect_id is not None:
            self.image_canvas.delete(self.roi_rect_id)
            self.roi_rect_id = None

        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        if right - left < 2 or bottom - top < 2:
            return

        active_data = self._get_active_data()
        src_h, src_w = active_data.shape
        x0i = int(np.clip(np.floor((left - self.image_offset_x) / self.zoom_scale), 0, src_w - 1))
        x1i = int(np.clip(np.ceil((right - self.image_offset_x) / self.zoom_scale), 0, src_w))
        y0i = int(np.clip(np.floor((top - self.image_offset_y) / self.zoom_scale), 0, src_h - 1))
        y1i = int(np.clip(np.ceil((bottom - self.image_offset_y) / self.zoom_scale), 0, src_h))
        if x1i <= x0i or y1i <= y0i:
            return

        roi = active_data[y0i:y1i, x0i:x1i]
        if roi.size == 0:
            return
        roi_low = float(np.percentile(roi, 1.0))
        roi_high = float(np.percentile(roi, 99.0))
        if roi_high <= roi_low:
            return
        self._update_contrast_slider_range()
        self.contrast_min_var.set(roi_low)
        self.contrast_max_var.set(roi_high)
        self.update_preview()

    def on_mouse_wheel(self, event: tk.Event[tk.Misc]) -> None:
        if getattr(event, "num", None) == 4:
            factor = 1.1
        elif getattr(event, "num", None) == 5:
            factor = 0.9
        else:
            delta = getattr(event, "delta", 0)
            factor = 1.1 if delta > 0 else 0.9
        self.zoom_at_point(float(event.x), float(event.y), factor)

    def _get_preview_save_folder(self) -> Path:
        if self.preview_save_folder is None:
            self.preview_save_folder = make_output_folder(self.source_path.parent)
        return self.preview_save_folder

    def _compute_rotational_profile(self, fft_data: np.ndarray) -> np.ndarray:
        if fft_data.size == 0:
            return np.zeros((1,), dtype=np.float32)
        rows, cols = fft_data.shape
        cy = rows / 2.0
        cx = cols / 2.0
        yy, xx = np.indices((rows, cols), dtype=np.float32)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r_bin = np.floor(rr).astype(np.int32)
        max_r = int(np.max(r_bin))
        radial_sum = np.bincount(r_bin.ravel(), weights=fft_data.ravel(), minlength=max_r + 1)
        radial_count = np.bincount(r_bin.ravel(), minlength=max_r + 1)
        valid = radial_count > 0
        profile = np.zeros((max_r + 1,), dtype=np.float32)
        profile[valid] = (radial_sum[valid] / radial_count[valid]).astype(np.float32)
        return profile

    def _build_rotational_average_image(self, fft_data: np.ndarray) -> np.ndarray:
        if fft_data.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
        profile = self._compute_rotational_profile(fft_data)
        rows, cols = fft_data.shape
        cy = rows / 2.0
        cx = cols / 2.0
        yy, xx = np.indices((rows, cols), dtype=np.float32)
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr = np.clip(rr, 0.0, float(profile.size - 1))
        r0 = np.floor(rr).astype(np.int32)
        r1 = np.clip(r0 + 1, 0, profile.size - 1)
        t = rr - r0.astype(np.float32)
        # Rotational average from 0°..180° (centrosymmetric with real-image FFTs).
        rotavg_img = (1.0 - t) * profile[r0] + t * profile[r1]
        return rotavg_img.astype(np.float32)

    def toggle_rotational_average_view(self) -> None:
        if self.current_fft_data.size == 0:
            messagebox.showwarning("Rotational average", "FFT data is not available.")
            return
        if self.display_mode != "fft":
            self.display_mode = "fft"
            self.fft_button.configure(text="Image")
        self.fft_pair_mode = not self.fft_pair_mode
        self.rotavg_button.configure(text="FFT only" if self.fft_pair_mode else "Rotational average")
        self._set_mode_label()
        self.auto_contrast()

    def save_current_image(self) -> None:
        save_format = self.parent.save_format_var.get()
        bit_depth = self.parent.bit_depth_var.get()
        if save_format in {"PNG", "JPEG"}:
            bit_depth = "8-bit"

        effective_format = "TIFF" if bit_depth == "16-bit" else save_format
        extension_by_format = {"TIFF": ".tif", "PNG": ".png", "JPEG": ".jpg"}
        extension = extension_by_format[effective_format]
        try:
            cutoff_value = float(self.cutoff_var.get())
        except ValueError:
            cutoff_value = 0.0
        cutoff_tag = cutoff_label(cutoff_value)
        output_name = f"{self.source_path.stem}_{cutoff_tag}{extension}"
        output_path = self._get_preview_save_folder() / output_name

        save_processed_image(
            output_path=output_path,
            filtered_data=self.current_display_8bit.astype(np.float32),
            save_format=effective_format,
            bit_depth=bit_depth,
            metadata=self.metadata,
            pixel_size_angstrom=max(self.current_binned_pixel_size, 1e-9),
        )
        messagebox.showinfo("Saved", f"Saved: {output_path}")


class BatchFFTFilterGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Batch FFT Gaussian Filter")
        self.geometry("700x420")

        self.selected_folder: Optional[Path] = None
        self.folder_pixel_size_angstrom: Optional[float] = None
        self.current_pixel_size_angstrom: Optional[float] = None
        self.logo_tk: Optional[ImageTk.PhotoImage] = None

        self.folder_var = tk.StringVar(value="No folder selected")
        self.pixel_size_var = tk.StringVar(value="Pixel size (Header): N/A")
        self.pixel_size_input_var = tk.StringVar(value="")
        self.cutoff_var = tk.StringVar(value="20")
        self.bin_factor_var = tk.StringVar(value="2")
        self.binned_pixel_size_var = tk.StringVar(value="Binned pixel size: N/A")
        self.save_format_var = tk.StringVar(value="TIFF")
        self.bit_depth_var = tk.StringVar(value="8-bit")

        self._build_ui()
        self.save_format_var.trace_add("write", self._on_save_format_changed)
        self.bin_factor_var.trace_add("write", self._on_bin_factor_changed)

    def _build_ui(self) -> None:
        top = tk.Frame(self, padx=12, pady=12)
        top.pack(fill="x")

        button_row = tk.Frame(top)
        button_row.grid(row=0, column=0, sticky="w")
        tk.Button(button_row, text="Open folder", command=self.select_folder, width=14).pack(side="left")
        tk.Button(button_row, text="Open file", command=self.open_file_preview, width=14).pack(side="left", padx=(8, 0))
        self.logo_label = tk.Label(top)
        self.logo_label.grid(row=0, column=1, sticky="e")
        self._load_logo()

        tk.Label(top, textvariable=self.folder_var, anchor="w").grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        pixel_row = tk.Frame(top)
        pixel_row.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky="w")
        tk.Label(pixel_row, textvariable=self.pixel_size_var, anchor="w").pack(side="left")
        tk.Label(pixel_row, text="Manual pixel size (Å/px):").pack(side="left", padx=(12, 0))
        pixel_entry = tk.Entry(pixel_row, textvariable=self.pixel_size_input_var, width=8)
        pixel_entry.pack(side="left", padx=(8, 0))
        pixel_entry.bind("<Return>", lambda _: self._apply_manual_pixel_size())
        tk.Button(pixel_row, text="Use", command=self._apply_manual_pixel_size, width=8).pack(side="left", padx=(8, 0))

        cutoff_row = tk.Frame(top)
        cutoff_row.grid(row=3, column=0, columnspan=2, pady=(14, 0), sticky="w")
        tk.Label(cutoff_row, text="Cut-off (Å):").pack(side="left")
        tk.Entry(cutoff_row, textvariable=self.cutoff_var, width=10).pack(side="left", padx=(8, 0))
        tk.Label(cutoff_row, text="Bin:").pack(side="left", padx=(16, 0))
        tk.OptionMenu(cutoff_row, self.bin_factor_var, "1", "2", "4", "8").pack(side="left", padx=(8, 0))
        tk.Label(cutoff_row, textvariable=self.binned_pixel_size_var).pack(side="left", padx=(16, 0))

        output_row = tk.Frame(top)
        output_row.grid(row=4, column=0, columnspan=2, pady=(10, 0), sticky="w")
        tk.Label(output_row, text="Save format:").pack(side="left")
        tk.OptionMenu(output_row, self.save_format_var, "TIFF", "PNG", "JPEG").pack(side="left", padx=(8, 16))
        tk.Label(output_row, text="Bit depth:").pack(side="left")
        tk.OptionMenu(output_row, self.bit_depth_var, "8-bit", "16-bit").pack(side="left", padx=(8, 0))

        self.filter_button = tk.Button(top, text="Filter", command=self.start_filtering, width=12)
        self.filter_button.grid(row=5, column=0, pady=(14, 0), sticky="w")

        self.log_text = tk.Text(self, height=16, padx=8, pady=8, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def _load_logo(self) -> None:
        logo_path = Path(__file__).with_name("Logo_TEM2C.png")
        if not logo_path.exists():
            return
        try:
            with Image.open(logo_path) as logo:
                logo_image = logo.convert("RGBA")
                logo_image.thumbnail((180, 70), Image.LANCZOS)
                self.logo_tk = ImageTk.PhotoImage(logo_image)
            self.logo_label.configure(image=self.logo_tk)
            self.logo_label.image = self.logo_tk
        except Exception:
            self.logo_label.configure(text="")

    def _on_save_format_changed(self, *_: object) -> None:
        if self.save_format_var.get() in {"PNG", "JPEG"} and self.bit_depth_var.get() != "8-bit":
            self.bit_depth_var.set("8-bit")
            self.log("Save format PNG/JPEG selected: bit depth switched to 8-bit.")

    def _on_bin_factor_changed(self, *_: object) -> None:
        self._refresh_binned_pixel_size()

    def _apply_manual_pixel_size(self) -> None:
        try:
            pixel_size = float(self.pixel_size_input_var.get())
            if pixel_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid pixel size", "Manual pixel size must be a positive number (Å/px).")
            return
        self.current_pixel_size_angstrom = pixel_size
        self.pixel_size_var.set(f"Pixel size (Manual): {pixel_size:.4f} Å")
        self._refresh_binned_pixel_size()

    def _get_manual_pixel_size(self) -> Optional[float]:
        try:
            pixel_size = float(self.pixel_size_input_var.get())
            if pixel_size > 0:
                return pixel_size
        except ValueError:
            return None
        return None

    def _get_effective_pixel_size(self) -> float:
        manual_pixel_size = self._get_manual_pixel_size()
        if manual_pixel_size is not None:
            return manual_pixel_size
        return self.current_pixel_size_angstrom or self.folder_pixel_size_angstrom or 1.0

    def _refresh_binned_pixel_size(self) -> None:
        try:
            bin_factor = int(self.bin_factor_var.get())
        except ValueError:
            self.binned_pixel_size_var.set("Binned pixel size: N/A")
            return
        if bin_factor not in {1, 2, 4, 8}:
            self.binned_pixel_size_var.set("Binned pixel size: N/A")
            return

        pixel_size = self._get_effective_pixel_size()
        if pixel_size and pixel_size > 0:
            self.binned_pixel_size_var.set(f"Binned pixel size: {pixel_size * float(bin_factor):.4f} Å")
        else:
            self.binned_pixel_size_var.set("Binned pixel size: N/A")

    def _get_cutoff_and_bin_factor(self) -> tuple[float, int]:
        try:
            cutoff = float(self.cutoff_var.get())
            if cutoff <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Cut-off must be a positive number (Å).") from exc

        try:
            bin_factor = int(self.bin_factor_var.get())
            if bin_factor not in {1, 2, 4, 8}:
                raise ValueError
        except ValueError as exc:
            raise ValueError("Bin factor must be 1, 2, 4 or 8.") from exc
        return cutoff, bin_factor

    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return

        self.selected_folder = Path(folder)
        self.folder_var.set(str(self.selected_folder))
        self.folder_pixel_size_angstrom = None
        self.current_pixel_size_angstrom = None
        self.pixel_size_input_var.set("")

        files = list_image_files(self.selected_folder)
        if not files:
            self.pixel_size_var.set("Pixel size (Header): N/A")
            self.log("No supported image files found in the selected folder.")
            return

        for file_path in files:
            try:
                _, pixel_size, _ = read_image_and_pixel_size(file_path)
                if pixel_size and pixel_size > 0:
                    self.folder_pixel_size_angstrom = pixel_size
                    break
            except Exception:
                continue

        if self.folder_pixel_size_angstrom:
            self.current_pixel_size_angstrom = self.folder_pixel_size_angstrom
            self.pixel_size_var.set(f"Pixel size (Header): {self.folder_pixel_size_angstrom:.4f} Å")
            self.pixel_size_input_var.set(f"{self.folder_pixel_size_angstrom:.4f}")
        else:
            self.pixel_size_var.set("Pixel size (Header): N/A")
            self.log("Pixel size could not be read from headers. Fallback value 1.0 Å will be used.")
        self._refresh_binned_pixel_size()

    def start_filtering(self) -> None:
        if self.selected_folder is None:
            messagebox.showerror("No folder selected", "Please click Open and select a folder first.")
            return

        try:
            cutoff, bin_factor = self._get_cutoff_and_bin_factor()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        save_format = self.save_format_var.get()
        bit_depth = self.bit_depth_var.get()
        if save_format in {"PNG", "JPEG"} and bit_depth != "8-bit":
            bit_depth = "8-bit"
            self.bit_depth_var.set("8-bit")
        self.filter_button.configure(state="disabled")
        thread = threading.Thread(
            target=self._run_batch_filter,
            args=(cutoff, bin_factor, save_format, bit_depth),
            daemon=True,
        )
        thread.start()

    def open_file_preview(self) -> None:
        try:
            cutoff, bin_factor = self._get_cutoff_and_bin_factor()
        except ValueError as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return

        file_path_text = filedialog.askopenfilename(
            title="Select image file",
            filetypes=[
                ("Supported images", "*.mrc *.mrcs *.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path_text:
            return

        file_path = Path(file_path_text)
        try:
            data, per_file_pixel_size, metadata = read_image_and_pixel_size(file_path)
            gray = to_grayscale(data)
            if per_file_pixel_size and per_file_pixel_size > 0:
                self.current_pixel_size_angstrom = per_file_pixel_size
                self.pixel_size_var.set(f"Pixel size (Header): {per_file_pixel_size:.4f} Å ({file_path.name})")
                self.pixel_size_input_var.set(f"{per_file_pixel_size:.4f}")
            else:
                self.current_pixel_size_angstrom = self.folder_pixel_size_angstrom
                self.pixel_size_var.set("Pixel size (Header): N/A")
                self.log(f"Pixel size not found in header for {file_path.name}; fallback value 1.0 Å may be used.")
            self._refresh_binned_pixel_size()

            manual_pixel_size = self._get_manual_pixel_size()
            pixel_size = manual_pixel_size or per_file_pixel_size or self.folder_pixel_size_angstrom or 1.0
            SingleFilePreviewWindow(
                parent=self,
                source_path=file_path,
                source_gray=gray.astype(np.float32),
                metadata=metadata,
                cutoff_angstrom=cutoff,
                pixel_size_angstrom=pixel_size,
                bin_factor=bin_factor,
            )
        except Exception as exc:
            messagebox.showerror("Open file failed", str(exc))

    def _run_batch_filter(self, cutoff_angstrom: float, bin_factor: int, save_format: str, bit_depth: str) -> None:
        assert self.selected_folder is not None
        image_files = list_image_files(self.selected_folder)
        if not image_files:
            self.after(0, lambda: messagebox.showerror("No images", "No supported image files found."))
            self.after(0, lambda: self.filter_button.configure(state="normal"))
            return

        output_folder = make_output_folder(self.selected_folder)
        cutoff_tag = cutoff_label(cutoff_angstrom)

        effective_format = "TIFF" if bit_depth == "16-bit" else save_format
        extension_by_format = {"TIFF": ".tif", "PNG": ".png", "JPEG": ".jpg"}
        extension = extension_by_format[effective_format]

        self.after(0, lambda: self.log(f"Output folder: {output_folder}"))
        self.after(
            0,
            lambda: self.log(
                f"Processing {len(image_files)} files with cut-off = {cutoff_angstrom:.3f} Å and bin {bin_factor}"
            ),
        )
        processed = 0
        failed = 0

        for file_path in image_files:
            try:
                data, per_file_pixel_size, metadata = read_image_and_pixel_size(file_path)
                gray = to_grayscale(data)
                binned = bin_image_bilinear(gray, bin_factor)
                contrasted = auto_contrast_to_uint8(binned)

                manual_pixel_size = self._get_manual_pixel_size()
                pixel_size = manual_pixel_size or per_file_pixel_size or self.folder_pixel_size_angstrom or 1.0
                binned_pixel_size = pixel_size * float(bin_factor)
                filtered = gaussian_lowpass_fft(contrasted.astype(np.float32), binned_pixel_size, cutoff_angstrom)
                output_name = f"{file_path.stem}_{cutoff_tag}{extension}"
                output_path = output_folder / output_name

                save_processed_image(
                    output_path=output_path,
                    filtered_data=filtered,
                    save_format=effective_format,
                    bit_depth=bit_depth,
                    metadata=metadata,
                    pixel_size_angstrom=binned_pixel_size,
                )

                processed += 1
                self.after(0, lambda p=file_path.name, o=output_name: self.log(f"OK: {p} -> {o}"))
            except Exception as exc:
                failed += 1
                self.after(0, lambda p=file_path.name, e=str(exc): self.log(f"ERROR: {p} ({e})"))

        summary = f"Done. Success: {processed}, Failed: {failed}"
        self.after(0, lambda: self.log(summary))
        self.after(0, lambda: self.filter_button.configure(state="normal"))


def main() -> None:
    app = BatchFFTFilterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

"""Written by Denis Chrétien, using VS Code IDE and Tkinter library.
   Designed with AI assistance.
   Last updated: November 2026.
   Contact: denis.chretien@cnrs.fr"""