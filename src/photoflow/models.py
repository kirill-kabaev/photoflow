"""Versioned, closed contracts. Unknown fields and operations are errors."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,100}$")]
Version = Annotated[str, Field(pattern=r"^v[0-9]{3,}$")]
Checksum = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Text = Annotated[str, Field(min_length=1, max_length=20000)]
Unit = Annotated[float, Field(ge=0, le=1)]
State = Literal["pending", "analyzing", "ready", "processing", "review",
                "revision_requested", "approved", "exporting", "exported", "failed"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RawSettings(StrictModel):
    white_balance: Literal["camera", "daylight", "custom"] = "camera"
    multipliers: Annotated[list[Annotated[float, Field(gt=0, le=16)]], Field(min_length=4, max_length=4)] | None = None
    highlight_mode: Literal["clip", "blend", "reconstruct"] = "clip"

    @model_validator(mode="after")
    def custom_wb(self):
        if (self.white_balance == "custom") != (self.multipliers is not None):
            raise ValueError("custom WB requires exactly four multipliers; other modes forbid them")
        return self


class GlobalAdjustments(StrictModel):
    exposure_ev: Annotated[float, Field(ge=-3, le=3)] = 0.0
    highlights: Annotated[float, Field(ge=0, le=1)] = 0.0
    shadows: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0
    black_point: Annotated[float, Field(ge=-0.05, le=0.05)] = 0.0
    white_point: Annotated[float, Field(ge=0.8, le=1.2)] = 1.0
    contrast: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0
    saturation: Annotated[float, Field(ge=0, le=1.5)] = 1.0
    vibrance: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0
    channel_gains: Annotated[list[Annotated[float, Field(ge=0.75, le=1.25)]], Field(min_length=3, max_length=3)] = [1.0, 1.0, 1.0]
    tone_curve: Annotated[list[Annotated[list[Unit], Field(min_length=2, max_length=2)]], Field(min_length=2, max_length=16)] = [[0.0, 0.0], [1.0, 1.0]]
    sharpen: Annotated[float, Field(ge=0, le=0.5)] = 0.0
    denoise: Annotated[float, Field(ge=0, le=0.3)] = 0.0

    @model_validator(mode="after")
    def curve(self):
        points = self.tone_curve
        if points[0] != [0.0, 0.0] or points[-1] != [1.0, 1.0]:
            raise ValueError("tone_curve must start [0,0] and end [1,1]")
        if any(b[0] <= a[0] or b[1] < a[1] for a, b in zip(points, points[1:])):
            raise ValueError("tone_curve must be increasing in x, nondecreasing in y")
        return self


class Region(StrictModel):
    """AI-selected ellipse in oriented, uncropped normalized image coordinates."""
    id: Identifier
    kind: Literal["ellipse"] = "ellipse"
    center: Annotated[list[Unit], Field(min_length=2, max_length=2)]
    radius: Annotated[list[Annotated[float, Field(gt=0, le=1)]], Field(min_length=2, max_length=2)]
    feather: Annotated[float, Field(gt=0, le=1)] = 0.3
    description: Text


class LocalAdjustment(StrictModel):
    mask: Identifier
    operation: Literal["exposure", "redness", "saturation", "sharpen", "smooth"]
    amount: Annotated[float, Field(ge=-1, le=1)]
    rationale: Text

    @model_validator(mode="after")
    def safe_strength(self):
        bounds = {"exposure": (-0.75, 0.75), "redness": (0, 0.3),
                  "saturation": (-0.3, 0.3), "sharpen": (0, 0.5), "smooth": (0, 0.25)}
        lo, hi = bounds[self.operation]
        if not lo <= self.amount <= hi:
            raise ValueError(f"{self.operation}: amount must be in [{lo}, {hi}]")
        return self


class Crop(StrictModel):
    left: Unit = 0.0
    top: Unit = 0.0
    right: Unit = 1.0
    bottom: Unit = 1.0

    @model_validator(mode="after")
    def nonempty(self):
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("crop must have positive area")
        return self


class Output(StrictModel):
    jpeg_quality: Annotated[int, Field(ge=90, le=100)] = 95
    color_space: Literal["sRGB"] = "sRGB"
    tiff_bits: Literal[16] = 16


class Recipe(StrictModel):
    schema_version: Literal[1] = 1
    photo_id: Identifier
    source_sha256: Checksum
    parent_version: Version | None = None
    engine_version: Literal[__version__] = __version__
    style: Text
    raw: RawSettings = RawSettings()
    global_adjustments: GlobalAdjustments = GlobalAdjustments()
    masks: Annotated[list[Region], Field(max_length=32)] = []
    local_adjustments: Annotated[list[LocalAdjustment], Field(max_length=64)] = []
    crop: Crop = Crop()
    output: Output = Output()
    rationale: Text
    feedback_applied: str = ""
    feedback_sha256: Checksum | None = None
    unfulfilled_feedback: list[Text] = []

    @model_validator(mode="after")
    def mask_references(self):
        ids = [m.id for m in self.masks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate mask IDs")
        if any(a.mask not in ids for a in self.local_adjustments):
            raise ValueError("local adjustment references unknown mask")
        return self


class Analysis(StrictModel):
    schema_version: Literal[1] = 1
    photo_id: Identifier
    source_sha256: Checksum
    preview_sha256: Checksum
    provider: Literal["codex", "test_fixture"]
    model: Text
    visually_inspected: Literal[True]
    category: Literal["studio_portrait", "fashion", "beauty", "lifestyle", "product", "landscape", "other"]
    people_present: bool
    observations: Annotated[list[Text], Field(min_length=1)]
    technical: dict[str, Text]
    portrait: list[Text] = []
    recommendations: Annotated[list[Text], Field(min_length=1)]
    uncertainties: Annotated[list[Text], Field(min_length=1)]
    human_attention: list[Text] = []


class Plan(StrictModel):
    analysis: Analysis
    recipe: Recipe

    @model_validator(mode="after")
    def same_source(self):
        if (self.analysis.photo_id, self.analysis.source_sha256) != (self.recipe.photo_id, self.recipe.source_sha256):
            raise ValueError("analysis and recipe must reference the same photo and source")
        return self


class AIReview(StrictModel):
    schema_version: Literal[1] = 1
    photo_id: Identifier
    version: Version
    comparison_sha256: Checksum
    provider: Literal["codex", "test_fixture"]
    model: Text
    visually_inspected: Literal[True]
    improvements: list[Text]
    concerns: list[Text]
    recommendation: Literal["human_review", "revise", "manual_attention"]
    rationale: Text


class Photo(StrictModel):
    id: Identifier
    filename: str
    source_sha256: Checksum
    duplicate_of: Identifier | None = None
    state: State = "pending"
    versions: list[Version] = []
    current_version: Version | None = None
    approved_version: Version | None = None
    approved_recipe_sha256: Checksum | None = None
    approved_by: str | None = None
    approved_at: str | None = None
    operation_version: Version | None = None
    priority: int = 0
    error: str | None = None
    updated_at: str
    history: list[dict] = []

    @model_validator(mode="after")
    def integrity(self):
        if len(self.versions) != len(set(self.versions)):
            raise ValueError("duplicate versions")
        if self.current_version and self.current_version not in self.versions:
            raise ValueError("current version is not registered")
        if self.approved_version and self.approved_version not in self.versions:
            raise ValueError("approved version is not registered")
        if self.state in {"approved", "exporting", "exported"} and not all([
            self.approved_version, self.approved_recipe_sha256, self.approved_by, self.approved_at
        ]):
            raise ValueError("approval provenance missing")
        return self


class Queue(StrictModel):
    schema_version: Literal[1] = 1
    updated_at: str
    photos: list[Photo] = []

    @model_validator(mode="after")
    def unique(self):
        if len({p.id for p in self.photos}) != len(self.photos) or len({p.filename for p in self.photos}) != len(self.photos):
            raise ValueError("duplicate queue identities")
        return self


class SessionStyle(StrictModel):
    schema_version: Literal[1] = 1
    style: Text = "Выбрать после визуального анализа"
    exposure: str = "Сохранять детали светов и теней"
    white_balance: str = "Опираться на изображение и условия освещения"
    contrast: str = "Естественный"
    palette: str = "Естественные цвета"
    skin_retouching: str = "Минимальная локальная коррекция; сохранять текстуру"
    background: str = "По результатам анализа"
    sharpening: str = "Умеренная, без ореолов"
    export: Output = Output()
    reference_photo: Identifier | None = None
    reference_version: Version | None = None


class VersionManifest(StrictModel):
    schema_version: Literal[1] = 1
    status: Literal["complete"]
    version: Version
    source_filename: Text
    source_sha256: Checksum
    parent_version: Version | None
    recipe_sha256: Checksum
    engine_version: Literal[__version__]
    libraries: dict
    ai: dict[str, str]
    masks: list[Region]
    created_at: Text
    artifacts: dict[str, Checksum]
    mutable_artifacts: list[str]

    @model_validator(mode="after")
    def required_artifacts(self):
        required = {"recipe.json", "analysis.json", "decoder.json", "session_style.json",
                    "metrics.json", "before.jpg", "after.jpg", "before_after.jpg"}
        if not required <= self.artifacts.keys():
            raise ValueError("Version manifest is missing required artifacts")
        for mask in self.masks:
            if not {f"masks/{mask.id}.npy", f"masks/{mask.id}-overlay.jpg"} <= self.artifacts.keys():
                raise ValueError("Version manifest is missing mask artifacts")
        return self


class ExportManifest(StrictModel):
    schema_version: Literal[1] = 1
    photo_id: Identifier
    source_sha256: Checksum
    source_version: Version
    recipe_sha256: Checksum
    approved_by: Text
    approved_at: Text
    created_at: Text
    validation: dict
    libraries: dict
    files: dict[str, Checksum]

    @model_validator(mode="after")
    def required_files(self):
        if set(self.files) != {f"{self.photo_id}.jpg", f"{self.photo_id}.tif"}:
            raise ValueError("Export manifest must reference exactly the photo JPEG and TIFF")
        return self
