"""Validated, read-only linear RAW cache. Caller holds the session lock."""
import hashlib
import os
import uuid
from typing import Literal

import numpy as np

from . import __version__, imaging
from .models import Checksum, RawSettings, StrictModel
from .storage import FlowError, digest, json_bytes, read_model, sync_dir, write_json


class CacheEntry(StrictModel):
    schema_version: Literal[1] = 1
    key: Checksum
    source_sha256: Checksum
    settings: RawSettings
    libraries: dict
    decoder: str
    pixels_sha256: Checksum
    shape: list[int]
    metadata: dict


class RawCache:
    def __init__(self, session, decoder):
        self.session, self.decoder = session, decoder
        self.events = []

    def get(self, photo, settings):
        source = self.session.source(photo)  # Integrity is checked even on cache hits.
        decoder_id = f"{self.decoder.__module__}.{self.decoder.__qualname__}"
        specification = {"source_sha256": photo.source_sha256, "settings": settings.model_dump(),
                         "libraries": imaging.runtime_versions(), "decoder": decoder_id,
                         "engine_version": __version__}
        key = hashlib.sha256(json_bytes(specification)).hexdigest()
        folder = self.session.work(photo.id, "raw_cache", key)
        if folder.exists():
            entry = read_model(self.session.work(photo.id, "raw_cache", key, "entry.json"), CacheEntry)
            pixels_path = self.session.work(photo.id, "raw_cache", key, "pixels.npy")
            if (entry.key != key or entry.source_sha256 != photo.source_sha256
                    or entry.settings != settings or entry.libraries != specification["libraries"]
                    or entry.decoder != decoder_id or digest(pixels_path) != entry.pixels_sha256):
                raise FlowError("RAW cache integrity mismatch; inspect/quarantine cache explicitly")
            pixels = np.load(pixels_path, mmap_mode="r", allow_pickle=False)
            if (pixels.dtype != np.float32 or list(pixels.shape) != entry.shape
                    or pixels.ndim != 3 or pixels.shape[2] != 3):
                raise FlowError("RAW cache dtype or dimensions invalid")
            self.events.append({"key": key, "hit": True})
            return imaging.Decoded(pixels, entry.metadata)
        staging = self.session.work(photo.id, "raw_cache", f".staging-{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        decoded = self.decoder(source, settings)
        pixels = decoded.pixels
        if (pixels.dtype != np.float32 or pixels.ndim != 3 or pixels.shape[2] != 3
                or not np.isfinite(pixels).all() or min(pixels.shape[:2]) < 1):
            raise FlowError("Decoder returned invalid linear float32 RGB")
        path = staging / "pixels.npy"
        with path.open("xb") as handle:
            np.save(handle, pixels, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        self.session.source(photo)
        entry = CacheEntry(key=key, source_sha256=photo.source_sha256, settings=settings,
                           libraries=specification["libraries"], decoder=decoder_id,
                           pixels_sha256=digest(path), shape=list(pixels.shape), metadata=decoded.metadata)
        write_json(staging / "entry.json", entry)
        os.rename(staging, folder)
        sync_dir(folder.parent)
        self.events.append({"key": key, "hit": False})
        # Return a read-only memory map rather than retaining another full in-memory copy.
        return imaging.Decoded(np.load(folder / "pixels.npy", mmap_mode="r", allow_pickle=False), decoded.metadata)
