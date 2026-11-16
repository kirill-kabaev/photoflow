import numpy as np
import pytest
import tifffile
from PIL import Image
from pydantic import ValidationError

from photoflow import imaging
from photoflow.models import Recipe, Region
from photoflow.storage import FlowError


def recipe(**kwargs):
    return Recipe(photo_id="test", source_sha256="0" * 64, style="test", rationale="test", **kwargs)


def test_srgb_roundtrip_and_exposure(pixels):
    assert np.allclose(imaging.linearize_srgb(imaging.encode_srgb(pixels)), pixels, atol=2e-7)
    identity = imaging.process_pixels(pixels, recipe())
    assert identity.dtype == np.float32
    assert np.allclose(identity, pixels, atol=2e-7)
    from photoflow.models import GlobalAdjustments
    increased = imaging.process_pixels(pixels, recipe(global_adjustments=GlobalAdjustments(exposure_ev=1.0)))
    assert np.allclose(increased, pixels * 2, atol=3e-7)
    assert imaging.metrics(increased)["mean_luminance"] > imaging.metrics(pixels)["mean_luminance"]


def test_color_and_tone_are_recipe_controlled(pixels):
    from photoflow.models import GlobalAdjustments
    changed = imaging.process_pixels(pixels, recipe(global_adjustments=GlobalAdjustments(
        channel_gains=[1.1, 1.0, 0.9], contrast=0.1, highlights=0.2, shadows=0.1,
        saturation=0.9, vibrance=0.1, sharpen=0.1, denoise=0.05)))
    assert changed.shape == pixels.shape
    assert np.isfinite(changed).all()
    assert not np.allclose(changed, pixels)


def test_feathered_mask_local_exposure_and_overlay(pixels, tmp_path):
    from photoflow.models import LocalAdjustment
    region = Region(id="face", center=[0.5, 0.5], radius=[0.25, 0.3], feather=0.4, description="Synthetic region")
    mask = imaging.ellipse_mask(pixels.shape, region)
    assert mask.dtype == np.float32 and mask.min() == 0 and mask.max() == 1
    assert np.any((mask > 0) & (mask < 1))
    adjusted = imaging.process_pixels(pixels, recipe(masks=[region], local_adjustments=[
        LocalAdjustment(mask="face", operation="exposure", amount=0.5, rationale="test")]), tmp_path)
    assert np.allclose(adjusted[mask == 0], pixels[mask == 0], atol=3e-7)
    assert (adjusted[mask == 1] > pixels[mask == 1]).all()
    assert np.array_equal(np.load(tmp_path / "face.npy", allow_pickle=False), mask)
    with Image.open(tmp_path / "face-overlay.jpg") as overlay:
        assert overlay.size == (120, 80)


def test_redness_reduces_red_chroma_preserves_luminance(pixels):
    from photoflow.models import LocalAdjustment
    region = Region(id="nose", center=[0.5, 0.5], radius=[1.0, 1.0], description="test")
    changed = imaging.process_pixels(pixels, recipe(masks=[region], local_adjustments=[
        LocalAdjustment(mask="nose", operation="redness", amount=0.2, rationale="test")]))
    assert np.allclose(imaging.luminance(changed), imaging.luminance(pixels), atol=3e-7)
    assert (changed[..., 0] - changed[..., 1]).mean() < (pixels[..., 0] - pixels[..., 1]).mean()


def test_crop_comparison_and_true_16bit_export(pixels, tmp_path):
    from photoflow.models import Crop
    cropped = imaging.process_pixels(pixels, recipe(crop=Crop(left=0.1, top=0.1, right=0.9, bottom=0.9)))
    baseline = imaging.crop_image(pixels, Crop(left=0.1, top=0.1, right=0.9, bottom=0.9))
    meta = imaging.save_previews(baseline, cropped, tmp_path)
    w, h = meta["preview_dimensions"]
    assert meta["comparison_dimensions"] == [w * 2, h + 32]
    validation = imaging.export_pixels(cropped, tmp_path / "full.jpg", tmp_path / "full.tif", 95)
    assert validation["tiff_bits_per_channel"] == 16
    with tifffile.TiffFile(tmp_path / "full.tif") as tiff:
        assert tiff.pages[0].dtype == np.uint16
        assert tiff.pages[0].bitspersample == 16
        assert tiff.pages[0].tags[34675].value
    with Image.open(tmp_path / "full.jpg") as jpeg:
        assert jpeg.size == (w, h)
        assert jpeg.info["icc_profile"]
    # At least one sample must carry precision beyond a byte expanded to 16 bits.
    data = tifffile.imread(tmp_path / "full.tif")
    assert np.any(data % 257 != 0)


@pytest.mark.parametrize("bad", [
    {"shell": "rm -rf /"},
    {"global_adjustments": {"exposure_ev": 8}},
    {"global_adjustments": {"saturation": float("nan")}},
    {"masks": [{"id": "../evil", "center": [0.5, 0.5], "radius": [0.1, 0.1], "description": "test"}]},
    {"local_adjustments": [{"mask": "missing", "operation": "exposure", "amount": 0.1, "rationale": "test"}]},
    {"raw": {"white_balance": "custom"}},
    {"crop": {"left": 0.8, "right": 0.1}},
    {"global_adjustments": {"tone_curve": [[0, 0], [0.6, 0.8], [0.5, 0.9], [1, 1]]}},
    {"local_adjustments": [{"mask": "face", "operation": "reconstruct_face", "amount": 0.1, "rationale": "test"}]},
])
def test_unsafe_recipe_rejected(bad):
    with pytest.raises(ValidationError):
        recipe(**bad)


def test_icc_timestamp_variation_is_accepted_but_wrong_profile_rejected():
    changed = bytearray(imaging.srgb_profile())
    changed[24:36] = bytes.fromhex('07e500010001000000000000')
    imaging.validate_srgb_profile(bytes(changed))
    from PIL import ImageCms
    with pytest.raises(FlowError, match="not RGB"):
        imaging.validate_srgb_profile(ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes())
