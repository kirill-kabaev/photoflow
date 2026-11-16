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


SemanticName = Literal["person", "background", "face", "face_skin", "hair", "skin",
                       "upper_clothing", "lower_clothing", "dress", "eyes", "lips"]


class SemanticRegion(StrictModel):
    """Model-derived region, optionally intersected with an artistic ellipse."""
    id: Identifier
    kind: Literal["semantic"] = "semantic"
    region: SemanticName
    intersect_ellipse: Identifier | None = None
    description: Text


class SegmentationSettings(StrictModel):
    cache_key: Checksum
    model: Literal["schp-atr-bisenet-r18-v1"] = "schp-atr-bisenet-r18-v1"


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


class DetailRegion(StrictModel):
    """Unscaled inspection crop in oriented, uncropped source coordinates."""
    id: Identifier
    bounds: Crop
    description: Text


class SpotHealing(StrictModel):
    """Permission reference only; target geometry comes from the verified proposal."""
    request_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    decision_sha256: Checksum


class NaturalSkinRetouch(StrictModel):
    enabled: bool = True
    tone_evenness: Annotated[float, Field(ge=0, le=0.25)] = 0.12
    redness: Annotated[float, Field(ge=0, le=0.15)] = 0.04


class ApprovedCrop(StrictModel):
    request_id: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    decision_sha256: Checksum
    aspect_ratio: Literal['16:9','9:16']
    dimensions: Annotated[list[Annotated[int, Field(gt=0)]], Field(min_length=2,max_length=2)]
    source_box: Annotated[list[Annotated[int, Field(ge=0)]], Field(min_length=4,max_length=4)]

    @model_validator(mode='after')
    def exact_geometry(self):
        w,h=self.dimensions; x1,y1,x2,y2=self.source_box
        a,b=map(int,self.aspect_ratio.split(':'))
        if not (0<=x1<x2<=w and 0<=y1<y2<=h) or (x2-x1)*b != (y2-y1)*a:
            raise ValueError('Approved crop must fit oriented RAW and have the exact aspect ratio')
        return self


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
    semantic_masks: Annotated[list[SemanticRegion], Field(max_length=32)] = []
    segmentation: SegmentationSettings | None = None
    spot_healing: SpotHealing | None = None
    # None preserves legacy recipes. Newly imported/explored plans get enabled
    # defaults; explicit {enabled:false} is the reversible opt-out.
    natural_skin: NaturalSkinRetouch | None = None
    local_adjustments: Annotated[list[LocalAdjustment], Field(max_length=64)] = []
    detail_regions: Annotated[list[DetailRegion], Field(max_length=12)] = []
    crop: Crop = Crop()
    approved_crop: ApprovedCrop | None = None
    output: Output = Output()
    rationale: Text
    feedback_applied: str = ""
    feedback_sha256: Checksum | None = None
    unfulfilled_feedback: list[Text] = []

    @model_validator(mode="after")
    def mask_references(self):
        ellipse_ids = [m.id for m in self.masks]
        ids = ellipse_ids + [m.id for m in self.semantic_masks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate mask IDs")
        if any(a.mask not in ids for a in self.local_adjustments):
            raise ValueError("local adjustment references unknown mask")
        if bool(self.semantic_masks) != (self.segmentation is not None):
            raise ValueError("semantic masks require an explicit segmentation cache binding")
        if any(m.intersect_ellipse and m.intersect_ellipse not in ellipse_ids for m in self.semantic_masks):
            raise ValueError("semantic intersection references unknown ellipse")
        detail_ids = [region.id for region in self.detail_regions]
        if len(set(detail_ids)) != len(detail_ids):
            raise ValueError("duplicate detail region IDs")
        for region in self.detail_regions:
            b, c = region.bounds, self.crop
            if b.left < c.left or b.top < c.top or b.right > c.right or b.bottom > c.bottom:
                raise ValueError("detail region must be contained in the output crop")
        if self.approved_crop:
            w,h=self.approved_crop.dimensions
            expected=[p/d for p,d in zip(self.approved_crop.source_box,[w,h,w,h])]
            if any(abs(a-b)>1e-12 for a,b in zip(expected,[self.crop.left,self.crop.top,self.crop.right,self.crop.bottom])):
                raise ValueError('Normalized crop differs from approved pixel coordinates')
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
    coverage: dict[str, Text] = {}


class Plan(StrictModel):
    analysis: Analysis
    recipe: Recipe

    @model_validator(mode="after")
    def same_source(self):
        if (self.analysis.photo_id, self.analysis.source_sha256) != (self.recipe.photo_id, self.recipe.source_sha256):
            raise ValueError("analysis and recipe must reference the same photo and source")
        return self


class CorrectionRecommendation(StrictModel):
    problem: Text
    region: Text
    proposed_correction: Text
    expected_improvement: Text
    side_effects: list[Text]
    verification: Text
    adjustment_paths: Annotated[list[Annotated[str, Field(pattern=r"^(?:global_adjustments\.(?:exposure_ev|highlights|shadows|black_point|white_point|contrast|saturation|vibrance|channel_gains|tone_curve|sharpen|denoise)|raw\.(?:white_balance|multipliers|highlight_mode)|masks|semantic_masks|segmentation|local_adjustments|crop|approved_crop|spot_healing|natural_skin)$")]], Field(min_length=1)]


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
    assessment: dict[str, Text] = {}
    inspected_artifacts: dict[str, Checksum] = {}
    full_resolution_texture_inspected: bool = False
    corrections: list[CorrectionRecommendation] = []
    best_version: Version | None = None
    selection_rationale: str = ""
    intervention: Literal["none", "artistic_ambiguity", "identity_risk", "missing_capability", "unresolved_defect"] = "none"


class RenderAttempt(StrictModel):
    version: Version
    recipe_sha256: Checksum
    started_at: str
    status: Literal["reserved", "complete", "failed"] = "reserved"
    reason: str | None = None


class ProcessingCycle(StrictModel):
    start_attempt: Annotated[int, Field(ge=0)]
    max_full_passes: Literal[1, 2]
    authorized_by: Text
    reason: Text
    created_at: Text


class RevisionRequest(StrictModel):
    parent_version: Version
    feedback_path: Annotated[str, Field(pattern=r"^(?:v[0-9]{3,}/feedback\.md|revision_requests/[0-9a-f]{32}\.md)$")]
    requested_at: str


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
    revision_request: RevisionRequest | None = None
    max_full_passes: Literal[1, 2] = 2
    render_attempts: list[RenderAttempt] = []
    processing_cycles: list[ProcessingCycle] = []
    selected_version: Version | None = None
    selection_rationale: str = ""
    directed: bool = False
    target_style: str | None = None
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
        if self.selected_version and self.selected_version not in self.versions:
            raise ValueError("selected version is not registered")
        if self.processing_cycles:
            starts = [cycle.start_attempt for cycle in self.processing_cycles]
            if starts != sorted(set(starts)) or starts[0] < 1 or starts[-1] > len(self.render_attempts):
                raise ValueError("invalid processing cycle audit boundaries")
            boundaries = [0, *starts, len(self.render_attempts)]
            limits = [self.max_full_passes, *[c.max_full_passes for c in self.processing_cycles]]
            if any(b-a > limit for a,b,limit in zip(boundaries,boundaries[1:],limits)):
                raise ValueError("full-resolution pass limit exceeded")
        elif len(self.render_attempts) > self.max_full_passes:
            raise ValueError("full-resolution pass limit exceeded")
        if len({a.version for a in self.render_attempts}) != len(self.render_attempts):
            raise ValueError("duplicate render attempts")
        if self.approved_version and self.approved_version not in self.versions:
            raise ValueError("approved version is not registered")
        if self.revision_request and self.revision_request.parent_version not in self.versions:
            raise ValueError("revision parent is not registered")
        if self.state in {"approved", "exporting", "exported"} and not all([
            self.approved_version, self.approved_recipe_sha256, self.approved_by, self.approved_at
        ]):
            raise ValueError("approval provenance missing")
        return self

    @property
    def active_attempt_count(self):
        return len(self.render_attempts) - (self.processing_cycles[-1].start_attempt if self.processing_cycles else 0)

    @property
    def active_pass_limit(self):
        return self.processing_cycles[-1].max_full_passes if self.processing_cycles else self.max_full_passes


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
    semantic_masks: list[SemanticRegion] = []
    segmentation: SegmentationSettings | None = None
    spot_healing: SpotHealing | None = None
    natural_skin: NaturalSkinRetouch | None = None
    approved_crop: ApprovedCrop | None = None
    detail_regions: list[DetailRegion] = []
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
        if self.semantic_masks:
            if not self.segmentation or "segmentation.json" not in self.artifacts:
                raise ValueError("Version manifest is missing segmentation provenance")
            for mask in self.semantic_masks:
                if f"masks/{mask.id}-overlay.jpg" not in self.artifacts:
                    raise ValueError("Version manifest is missing semantic mask overlay")
        if self.spot_healing and not {f"healing/{name}" for name in
                ["before.jpg", "after.jpg", "before_after.jpg", "operation.json", "proposal.json", "decision.json"]} <= self.artifacts.keys():
            raise ValueError("Version manifest is missing healing consent or native comparison")
        if self.approved_crop and not {'cropping/proposal.json','cropping/decision.json'} <= self.artifacts.keys():
            raise ValueError('Version manifest is missing crop approval')
        for region in self.detail_regions:
            required_detail = {f"details/{region.id}/{name}" for name in
                               ["before.jpg", "after.jpg", "before_after.jpg", "region.json"]}
            if not required_detail <= self.artifacts.keys():
                raise ValueError("Version manifest is missing detail artifacts")
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
