"""Rendering invariants and failure boundaries, beyond identity-image smoke tests."""
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from photoflow import imaging
from photoflow.models import Crop, DetailRegion, GlobalAdjustments, Recipe
from photoflow.storage import FlowError


def make_recipe(**changes):
    return Recipe(photo_id="test", source_sha256="0" * 64, style="test", rationale="Test rendering invariants", **changes)


@pytest.mark.parametrize("parameters", [
    {"exposure_ev": -3.0}, {"exposure_ev": 3.0},
    {"highlights": 1.0}, {"shadows": 0.5}, {"shadows": -0.5},
    {"contrast": 0.5}, {"contrast": -0.5},
    {"black_point": 0.05, "white_point": 0.8},
    {"black_point": -0.05, "white_point": 1.2},
    {"tone_curve": [[0, 0], [0.1, 0.03], [0.4, 0.5], [0.8, 0.9], [1, 1]]},
    {"exposure_ev": 1.0, "highlights": 0.7, "shadows": 0.3, "contrast": 0.2,
     "tone_curve": [[0, 0], [0.2, 0.25], [0.6, 0.75], [1, 1]]},
])
def test_global_tones_are_monotonic_neutral_and_bounded(parameters):
    ramp = np.repeat(np.linspace(0, 1, 4096, dtype=np.float32)[None, :, None], 3, axis=2)
    result = imaging.process_pixels(ramp, make_recipe(global_adjustments=GlobalAdjustments(**parameters)))
    assert np.isfinite(result).all() and result.dtype == np.float32
    assert result.min() >= 0 and result.max() <= 1
    assert np.min(np.diff(result[0, :, 0])) >= -2e-7
    assert np.allclose(result[..., 0], result[..., 1], atol=2e-7)
    assert np.allclose(result[..., 1], result[..., 2], atol=2e-7)


def test_exposure_precedes_display_curve_without_double_gamma():
    # sRGB display 0.5 is linear 0.2140411405. One EV -> linear 0.428082281.
    # sRGB(0.428082281) = 0.685836119; curve (0.5,0.6)..(1,1) maps to 0.748668895.
    source = np.full((1, 1, 3), 0.2140411405, dtype=np.float32)
    recipe = make_recipe(global_adjustments=GlobalAdjustments(
        exposure_ev=1.0, tone_curve=[[0, 0], [0.5, 0.6], [1, 1]]))
    display = imaging.encode_srgb(imaging.process_pixels(source, recipe))
    assert np.allclose(display, 0.748668895, atol=3e-7)
    # An already-encoded input accidentally treated as linear would be much brighter.
    assert not np.allclose(display, imaging.encode_srgb(source * 4), atol=0.01)


def test_highlight_compression_preserves_unclipped_rgb_ratios():
    source = np.array([[[1.0, 0.8, 0.6]]], dtype=np.float32)
    result = imaging.process_pixels(source, make_recipe(global_adjustments=GlobalAdjustments(highlights=0.8)))
    assert np.allclose(result[..., 0] / result[..., 1], source[..., 0] / source[..., 1], atol=3e-7)
    assert np.allclose(result[..., 1] / result[..., 2], source[..., 1] / source[..., 2], atol=3e-7)
    assert (result < source).all()


def test_display_clipping_is_explicit_and_precurve_compression_can_prevent_it():
    source = np.repeat(np.array([[[0.6], [0.8]]], dtype=np.float32), 3, axis=2)
    clipped = imaging.process_pixels(source, make_recipe(global_adjustments=GlobalAdjustments(exposure_ev=1.0)))
    assert np.allclose(clipped, 1.0)
    assert imaging.metrics(clipped)["highlight_clipped_pixel_fraction"] == 1.0
    protected = imaging.process_pixels(source, make_recipe(global_adjustments=GlobalAdjustments(exposure_ev=1.0, highlights=1.0)))
    assert np.all(protected < 1.0)
    assert protected[0, 0, 0] < protected[0, 1, 0]
    assert imaging.metrics(protected)["highlight_clipped_pixel_fraction"] == 0.0


def test_native_detail_coordinates_survive_output_crop(tmp_path):
    y, x = np.mgrid[:100, :200]
    source = np.stack([x / 300, y / 200, np.full_like(x, 0.1, dtype=float)], axis=-1).astype(np.float32)
    recipe = make_recipe(crop=Crop(left=0.1, top=0.1, right=0.9, bottom=0.9), detail_regions=[
        DetailRegion(id="face", bounds=Crop(left=0.25, top=0.2, right=0.5, bottom=0.6), description="Known test rectangle")])
    cropped = imaging.crop_image(source, recipe.crop)
    details = imaging.save_details(cropped, cropped, source.shape, recipe, tmp_path)
    assert details["face"]["source_box"] == [50, 20, 100, 60]
    assert details["face"]["preview_dimensions"] == [50, 40]
    assert details["face"]["scale"] == 1.0
    with Image.open(tmp_path / "details/face/after.jpg") as image:
        assert image.size == (50, 40)
        expected = np.asarray(imaging.to_preview(source[20:60, 50:100]))
        assert np.mean(np.abs(np.asarray(image).astype(float) - expected.astype(float))) < 2.0
    with Image.open(tmp_path / "details/face/before_after.jpg") as image:
        assert image.size == (100, 72)


def test_detail_outside_crop_or_empty_at_pixel_scale_rejected(tmp_path):
    with pytest.raises(ValidationError, match="contained"):
        make_recipe(crop=Crop(left=0.5), detail_regions=[
            DetailRegion(id="bad", bounds=Crop(left=0.1, right=0.6), description="Invalid bounds")])
    recipe = make_recipe(detail_regions=[DetailRegion(id="tiny", bounds=Crop(right=0.00001), description="Subpixel")])
    pixels = np.zeros((10, 10, 3), dtype=np.float32)
    with pytest.raises(FlowError, match="detail crop"):
        imaging.save_details(pixels, pixels, pixels.shape, recipe, tmp_path)
