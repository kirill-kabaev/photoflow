import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photoflow import imaging
from photoflow.models import RawSettings, Recipe
from photoflow.storage import digest


@pytest.mark.raw
def test_real_raw_decode_process_export_and_immutability(tmp_path):
    sample = os.environ.get("PHOTOFLOW_RAW_SAMPLE")
    if not sample:
        pytest.skip("Set PHOTOFLOW_RAW_SAMPLE for separate real LibRaw integration test")
    path = Path(sample)
    before = digest(path)
    decoded = imaging.decode_raw(path, RawSettings())
    assert decoded.pixels.dtype == np.float32
    assert min(decoded.pixels.shape[:2]) > 1000
    # Orientation is delegated to LibRaw, metadata records oriented output dimensions.
    assert decoded.metadata["oriented_dimensions"] == [decoded.pixels.shape[1], decoded.pixels.shape[0]]
    active = decoded.metadata["active_dimensions_before_orientation"]
    expected = list(reversed(active)) if decoded.metadata["orientation_flip"] in {5, 6} else active
    assert decoded.metadata["oriented_dimensions"] == expected
    recipe = Recipe(photo_id="raw-integration", source_sha256=before, style="neutral",
                    rationale="Real RAW integration test, no AI or artistic claims")
    processed = imaging.process_pixels(decoded.pixels, recipe)
    assert np.allclose(processed, decoded.pixels, atol=4e-7)
    imaging.save_previews(decoded.pixels, processed, tmp_path)
    result = imaging.export_pixels(processed, tmp_path / "output.jpg", tmp_path / "output.tif", 95)
    assert result["tiff_bits_per_channel"] == 16
    assert digest(path) == before
