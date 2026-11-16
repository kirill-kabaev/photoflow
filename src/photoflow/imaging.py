"""High precision deterministic development. All artistic parameters are explicit."""
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import rawpy
import scipy
import tifffile
from PIL import Image, ImageCms, ImageDraw, __version__ as pillow_version
from scipy.ndimage import gaussian_filter

from .models import RawSettings, Recipe
from .storage import FlowError, write_json


@dataclass
class Decoded:
    pixels: np.ndarray  # oriented, linear-light sRGB float32, nominal range [0, 1]
    metadata: dict


def runtime_versions():
    return {"numpy": np.__version__, "rawpy": rawpy.__version__,
            "libraw": list(rawpy.libraw_version), "pillow": pillow_version,
            "scipy": scipy.__version__, "tifffile": tifffile.__version__}


def decode_raw(path: Path, settings: RawSettings):
    try:
        with rawpy.imread(str(path)) as raw:
            modes = {"clip": rawpy.HighlightMode.Clip, "blend": rawpy.HighlightMode.Blend,
                     "reconstruct": rawpy.HighlightMode.ReconstructDefault}
            image = raw.postprocess(
                output_bps=16, output_color=rawpy.ColorSpace.sRGB, gamma=(1, 1),
                no_auto_bright=True, bright=1.0,
                use_camera_wb=settings.white_balance == "camera", use_auto_wb=False,
                user_wb=settings.multipliers,
                highlight_mode=modes[settings.highlight_mode],
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
            )
            metadata = {"oriented_dimensions": [image.shape[1], image.shape[0]],
                        "active_dimensions_before_orientation": [raw.sizes.width, raw.sizes.height],
                        "sensor_dimensions": [raw.sizes.raw_width, raw.sizes.raw_height],
                        "orientation_flip": raw.sizes.flip,
                        "camera_white_balance": list(raw.camera_whitebalance),
                        "black_levels": list(raw.black_level_per_channel),
                        "white_level": raw.white_level,
                        "working_space": "linear sRGB", "decoder_bits": 16,
                        "raw_settings": settings.model_dump(), "libraries": runtime_versions()}
    except rawpy.LibRawError as exc:
        raise FlowError(f"LibRaw cannot decode {path.name}: {exc}") from exc
    return Decoded(image.astype(np.float32) / np.float32(65535), metadata)


def encode_srgb(image):
    x = np.clip(image, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def linearize_srgb(image):
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4).astype(np.float32)


@lru_cache(maxsize=1)
def srgb_profile():
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def validate_srgb_profile(data):
    profile = ImageCms.ImageCmsProfile(BytesIO(data))
    if profile.profile.xcolor_space.strip() != "RGB":
        raise FlowError("ICC profile is not RGB")
    # LittleCMS stamps creation time and may derive the profile ID from it.
    def canonical(value):
        normalized = bytearray(value)
        normalized[24:36] = bytes(12)
        normalized[84:100] = bytes(16)
        return normalized
    if canonical(data) != canonical(srgb_profile()):
        raise FlowError("Output ICC profile does not describe the expected sRGB encoding")


def luminance(image):
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def metrics(image):
    y = luminance(image)
    return {"dimensions": [image.shape[1], image.shape[0]], "dtype": str(image.dtype),
            "working_space": "linear sRGB", "min": float(image.min()), "max": float(image.max()),
            "mean_luminance": float(y.mean()),
            "luminance_percentiles": [float(v) for v in np.percentile(y, [1, 5, 50, 95, 99])],
            "highlight_clipped_pixel_fraction": float(np.any(image >= 1, axis=2).mean()),
            "shadow_clipped_pixel_fraction": float(np.all(image <= 0, axis=2).mean())}


def ellipse_mask(shape, region):
    h, w = shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = region.center
    rx, ry = region.radius
    distance = np.sqrt(((x + 0.5) / w - cx) ** 2 / rx**2 + ((y + 0.5) / h - cy) ** 2 / ry**2)
    # Smoothstep ramp fully inside the selected ellipse, no hard edge.
    t = np.clip((1 - distance) / region.feather, 0, 1)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _blur(image, sigma=1.0):
    return gaussian_filter(image, sigma=(sigma, sigma, 0), mode="reflect")


def _smooth(image, amount, sigma=1.0):
    # Edge-weighted low-amplitude denoise, not a semantic skin model.
    base = _blur(image, sigma)
    difference = np.mean((image - base) ** 2, axis=2, keepdims=True)
    edge_weight = np.exp(-difference / 0.0004)
    return image + amount * edge_weight * (base - image)


def crop_image(image, crop):
    h, w = image.shape[:2]
    x1, y1 = int(crop.left * w), int(crop.top * h)
    x2, y2 = int(crop.right * w), int(crop.bottom * h)
    if x2 <= x1 or y2 <= y1:
        raise FlowError("Crop is empty at source resolution")
    return image[y1:y2, x1:x2].copy()


def _global_pixels(source, a, spatial_scale=1.0):
    image = source.copy()
    image *= np.float32(2**a.exposure_ev)
    image *= np.asarray(a.channel_gains, dtype=np.float32)
    # Compression is luminance-linked, preserves ratios; cannot restore clipped RAW detail.
    if a.highlights:
        y = luminance(image)
        mapped = y / (1 + a.highlights * np.maximum(y - 0.5, 0))
        image *= (mapped / np.maximum(y, 1e-8))[..., None]
    if a.shadows:
        y = luminance(image)
        image *= np.power(2.0, a.shadows * (1 - np.clip(y / 0.25, 0, 1)))[..., None]
    image = (image - a.black_point) / (a.white_point - a.black_point)
    # Display-referred operations are explicitly applied after the sRGB transfer function.
    display = encode_srgb(image)
    display = (display - 0.5) * (1 + a.contrast) + 0.5
    display = np.interp(display, *np.asarray(a.tone_curve).T).astype(np.float32)
    gray = luminance(display)[..., None]
    chroma = display - gray
    approx_saturation = display.max(axis=2) - display.min(axis=2)
    multiplier = a.saturation + a.vibrance * (1 - approx_saturation)
    display = gray + chroma * multiplier[..., None]
    if a.denoise:
        display = _smooth(display, a.denoise, spatial_scale)
    if a.sharpen:
        display += a.sharpen * np.clip(display - _blur(display, spatial_scale), -0.08, 0.08)
    image = linearize_srgb(np.clip(display, 0, 1))
    return image


def process_pixels(source, recipe: Recipe, mask_dir=None, preview_size=1400, spatial_scale=1.0):
    if not 0 < spatial_scale <= 1:
        raise FlowError("Spatial preview scale must be in (0, 1]")
    if source.dtype != np.float32 or source.ndim != 3 or source.shape[2] != 3 or not np.isfinite(source).all():
        raise FlowError("Engine expects finite float32 RGB pixels")
    image = np.empty_like(source, subok=False)
    # All global operators are pointwise except sigma-scaled Gaussian filters.
    # Two sequential filters require at most twice their four-sigma radius.
    halo = 2 * int(np.ceil(4 * spatial_scale))
    for first in range(0, source.shape[0], 256):
        last = min(first + 256, source.shape[0])
        top, bottom = max(0, first - halo), min(source.shape[0], last + halo)
        tile = _global_pixels(source[top:bottom], recipe.global_adjustments, spatial_scale)
        image[first:last] = tile[first - top:last - top]
        del tile
    preview = to_preview(source, preview_size) if mask_dir is not None and recipe.masks else None
    # Generate one full-resolution mask at a time to bound peak memory.
    for region in recipe.masks:
        mask = ellipse_mask(image.shape, region)
        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
            np.save(mask_dir / f"{region.id}.npy", mask, allow_pickle=False)
            small = Image.fromarray(np.uint8(np.rint(mask * 255))).resize(preview.size, Image.Resampling.BILINEAR)
            overlay = Image.new("RGB", preview.size, (255, 60, 0))
            alpha = small.point(lambda value: int(value * 0.4))
            Image.composite(overlay, preview, alpha).save(mask_dir / f"{region.id}-overlay.jpg", icc_profile=srgb_profile(), quality=92)
        for operation in [op for op in recipe.local_adjustments if op.mask == region.id]:
            amount = operation.amount
            if operation.operation == "exposure":
                image *= np.power(2.0, amount * mask)[..., None]
            elif operation.operation == "redness":
                # Luminance-preserving reduction of positive red-minus-green chroma.
                delta = np.maximum(image[..., 0] - image[..., 1], 0) * amount * mask
                image[..., 0] -= delta
                image[..., 1] += delta * (0.2126 / 0.7152)
            elif operation.operation == "saturation":
                gray = luminance(image)[..., None]
                image += (image - gray) * (amount * mask)[..., None]
            elif operation.operation == "sharpen":
                image += amount * mask[..., None] * np.clip(image - _blur(image, spatial_scale), -0.04, 0.04)
            elif operation.operation == "smooth":
                image += mask[..., None] * (_smooth(image, amount, spatial_scale) - image)
        del mask
    if not np.isfinite(image).all():
        raise FlowError("Processing produced nonfinite pixels")
    return crop_image(image.astype(np.float32, copy=False), recipe.crop)


def to_preview(linear, max_edge=1400):
    # Resize float channels before quantization, preserving high precision until display.
    h, w = linear.shape[:2]
    scale = min(1, max_edge / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = np.stack([np.asarray(Image.fromarray(linear[..., c]).resize(size, Image.Resampling.LANCZOS))
                      for c in range(3)], axis=-1)
    return Image.fromarray(np.uint8(np.rint(encode_srgb(small) * 255)))


def save_previews(before, after, folder, max_edge=1400):
    if before.shape != after.shape:
        raise FlowError("Before/after dimensions differ")
    left, right = to_preview(before, max_edge), to_preview(after, max_edge)
    left.save(folder / "before.jpg", quality=93, icc_profile=srgb_profile())
    right.save(folder / "after.jpg", quality=93, icc_profile=srgb_profile())
    combined = Image.new("RGB", (left.width * 2, left.height + 32), (25, 25, 25))
    combined.paste(left, (0, 32))
    combined.paste(right, (left.width, 32))
    draw = ImageDraw.Draw(combined)
    draw.text((10, 8), "BEFORE", fill="white")
    draw.text((left.width + 10, 8), "AFTER", fill="white")
    combined.save(folder / "before_after.jpg", quality=93, icc_profile=srgb_profile())
    for name in ["before.jpg", "after.jpg", "before_after.jpg"]:
        with Image.open(folder / name) as test:
            test.verify()
    return {"preview_dimensions": list(left.size), "comparison_dimensions": list(combined.size)}


def save_details(before, after, source_shape, recipe, folder):
    """Native-resolution crops from processed float pixels, never from JPEG previews."""
    height, width = source_shape[:2]
    offset_x, offset_y = int(recipe.crop.left * width), int(recipe.crop.top * height)
    details = {}
    for region in recipe.detail_regions:
        b = region.bounds
        source_box = [int(b.left * width), int(b.top * height),
                      int(b.right * width), int(b.bottom * height)]
        x1, y1, x2, y2 = source_box
        x1, x2 = x1 - offset_x, x2 - offset_x
        y1, y2 = y1 - offset_y, y2 - offset_y
        if (x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0
                or x2 > after.shape[1] or y2 > after.shape[0] or before.shape != after.shape):
            raise FlowError(f"Invalid detail crop at source resolution: {region.id}")
        target = folder / "details" / region.id
        target.mkdir(parents=True, exist_ok=False)
        metadata = save_previews(before[y1:y2, x1:x2], after[y1:y2, x1:x2], target,
                                 max_edge=max(x2 - x1, y2 - y1))
        metadata.update(source_box=source_box, bounds=b.model_dump(), scale=1.0,
                        description=region.description, source="full-resolution linear RGB, before JPEG encoding")
        write_json(target / "region.json", metadata)
        details[region.id] = metadata
    return details


def export_pixels(linear, jpeg, tiff, quality):
    encoded = encode_srgb(linear)
    profile = srgb_profile()
    Image.fromarray(np.uint8(np.rint(encoded * 255))).save(
        jpeg, quality=quality, subsampling=0, icc_profile=profile)
    tifffile.imwrite(tiff, np.uint16(np.rint(encoded * 65535)), photometric="rgb",
                     metadata=None, compression=None, iccprofile=profile)
    return validate_export(jpeg, tiff, (linear.shape[1], linear.shape[0]))


def validate_export(jpeg, tiff, dimensions):
    with Image.open(jpeg) as image:
        image.load()
        if image.size != tuple(dimensions) or image.mode != "RGB" or not image.info.get("icc_profile"):
            raise FlowError("JPEG dimensions, mode or ICC profile invalid")
        validate_srgb_profile(image.info["icc_profile"])
    with tifffile.TiffFile(tiff) as file:
        page = file.pages[0]
        if page.dtype != np.dtype("uint16") or page.shape != (dimensions[1], dimensions[0], 3):
            raise FlowError("TIFF must be full-resolution RGB uint16")
        if page.photometric != tifffile.PHOTOMETRIC.RGB:
            raise FlowError("TIFF photometric interpretation is not RGB")
        if 34675 not in page.tags:
            raise FlowError("TIFF has no ICC profile")
        validate_srgb_profile(page.tags[34675].value)
    return {"dimensions": list(dimensions), "jpeg_bits": 8, "tiff_bits_per_channel": 16,
            "color_space": "sRGB", "icc_embedded": True, "tiff_compression": "none (lossless)"}
