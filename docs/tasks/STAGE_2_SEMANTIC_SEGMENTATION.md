# PhotoFlow — Stage 2: Local AI Semantic Segmentation

## Role

You are a Senior Computer Vision Engineer, Computational Photography Engineer, and Python Software Architect.

Your task is to integrate a **fully local AI semantic segmentation system** into the existing PhotoFlow RAW photography processing pipeline.

The objective is to replace inaccurate ellipse-based object masks with precise, image-aware semantic masks suitable for professional Natural Commercial Portrait processing.

The user must not manually draw masks, adjust segmentation boundaries, or specify object coordinates.

All segmentation inference must run locally without paid APIs or external image uploads.

## 1. Inspect the Existing Project

Before making changes, read:

- `AGENTS.md`
- `docs/MASTER_PLAN.md`
- `docs/ROADMAP.md`
- `docs/PROGRESS.md`, if available
- `docs/ai_workflow.md`
- `docs/processing_pipeline.md`
- `docs/portrait_rendering_diagnosis.md`

Inspect the existing:

- RAW decoding pipeline.
- Image orientation and coordinate conventions.
- Float32 processing buffers.
- Local mask implementation.
- Recipe schema.
- AI analysis and review contracts.
- Version management.
- Preview generation.
- Quality-control tests.
- Export pipeline.

Preserve existing functionality and backward compatibility where practical.

Do not rewrite the RAW engine unnecessarily.

## 2. Primary Objective

Implement local AI-powered semantic segmentation for portrait photographs.

Required semantic regions:

1. Person / foreground.
2. Background.
3. Face.
4. Facial skin.
5. Hair.
6. Visible skin, including hands and arms.
7. Upper clothing.
8. Lower clothing.

Support additional regions when available from the selected model.

Do not claim a region is supported unless the model actually predicts it or a separately validated algorithm derives it.

## 3. Model Selection

Evaluate suitable local segmentation models.

Start with:

- SCHP (Self-Correction for Human Parsing).
- A suitable modern human-parsing alternative, if available.
- Optional dedicated face-parsing models if required for accurate facial regions.

Consider:

- Segmentation accuracy.
- Available semantic classes.
- CPU inference performance.
- GPU acceleration support.
- Memory requirements.
- Model size.
- Licensing.
- Maintenance status.
- Python compatibility.
- PyTorch or ONNX Runtime integration.
- Reproducible model downloads.
- Suitability for portrait photography.

Prefer a model that distinguishes hair, facial skin, upper clothing, and lower clothing.

Do not choose a model solely because it is popular.

Do not assume that general person segmentation provides clothing-level masks.

### Model Selection Deliverable

Document:

- Models evaluated.
- Selected model.
- Available classes.
- Model version.
- Source and license.
- Required dependencies.
- CPU/GPU support.
- Known limitations.

Choose the smallest practical model that provides adequate quality on the actual PhotoFlow sample.

## 4. Local Execution Requirements

The segmentation system must:

- Run locally in WSL2 Ubuntu.
- Support CPU inference.
- Use GPU acceleration when available and correctly configured.
- Work without paid API access.
- Avoid mandatory cloud services.
- Avoid transmitting photographs externally.
- Run offline after model installation.

Detect available execution providers.

If GPU acceleration is unavailable, fall back to CPU.

Do not assume an NVIDIA GPU exists.

Do not modify system-wide CUDA drivers or WSL configuration automatically.

Keep segmentation dependencies isolated and reproducible.

Model weights must not be committed to Git.

Use a documented model-cache directory.

## 5. Segmentation Architecture

Implement a reusable segmentation subsystem.

Suggested responsibilities:

- Model loading.
- Image preprocessing.
- Inference.
- Semantic label mapping.
- Mask generation.
- Boundary refinement.
- Mask caching.
- Quality diagnostics.
- Integration with processing recipes.

Separate the segmentation model adapter from the rest of the RAW processing pipeline.

Use stable semantic region names independent of model-specific class IDs.

For example:

- `person`
- `background`
- `face`
- `face_skin`
- `hair`
- `skin`
- `upper_clothing`
- `lower_clothing`

Maintain an explicit mapping between model labels and PhotoFlow semantic regions.

Do not silently combine incompatible classes.

## 6. RAW Coordinate Alignment

This is a critical requirement.

The segmentation model will typically operate on a resized RGB preview, while PhotoFlow processes the original high-resolution RAW image.

Implement correct mapping between:

- Original sensor orientation.
- Oriented RAW coordinates.
- Segmentation input dimensions.
- Model output dimensions.
- Full-resolution processing coordinates.
- Optional crop coordinates.

Requirements:

- Preserve aspect ratio.
- Correctly handle letterboxing and padding.
- Avoid mask displacement.
- Avoid accidental horizontal or vertical flips.
- Handle portrait and landscape orientation.
- Handle non-square images.
- Respect the existing crop convention.

Use nearest-neighbor interpolation for discrete semantic labels.

Use suitable continuous interpolation for soft masks.

Add explicit alignment tests.

## 7. Mask Refinement

Raw neural-network predictions may have inaccurate boundaries.

Implement an optional image-guided refinement stage.

Possible approaches:

- Guided filtering.
- Edge-aware smoothing.
- Bilateral refinement.
- Morphological cleanup.
- Small-component removal.
- Controlled feathering.

Choose methods based on measured quality and computational cost.

Prioritize:

- Hairline boundaries.
- Loose hair strands.
- Shirt collar.
- Shirt sleeves.
- Shirt-to-trousers boundary.
- Hands and arms.
- Person-background silhouette.

Avoid excessive feathering that causes color spill.

Do not claim pixel-perfect hair segmentation.

Preserve natural transitions where appropriate.

## 8. Integration with RAW Processing

Extend the existing recipe schema to support semantic masks.

Example conceptual recipe:

```json
{
  "region": "face_skin",
  "operation": "exposure",
  "value_ev": 0.35
}
```

Adapt this example to the actual validated PhotoFlow schema.

The RAW engine must apply the requested operation only to the selected semantic region.

Support:

- Local exposure.
- Local saturation.
- Local color adjustments.
- Other existing compatible local operations.

Do not introduce arbitrary code execution through recipes.

### Important

Retain ellipse masks as optional artistic tools for broad dodge-and-burn lighting effects.

Use semantic masks as the preferred option for object-specific adjustments.

Allow semantic masks and elliptical masks to be combined when justified.

For example, an elliptical lighting gradient may be intersected with a face-skin mask to preserve soft lighting while preventing spill into hair or background.

Do not silently fall back to inaccurate ellipses when semantic segmentation fails.

## 9. Mask Caching

Segmentation should normally run once per photograph.

Cache generated masks using a key derived from:

- RAW file identity or checksum.
- Orientation.
- Segmentation model identifier.
- Model version or weights checksum.
- Preprocessing configuration.
- Mask refinement configuration.

Reuse masks across:

- Candidate recipes.
- Processing revisions.
- AI review cycles.
- Preview generation.
- Full-resolution rendering.

Invalidate cached masks when relevant inputs change.

Write cache files atomically.

Avoid storing duplicate full-resolution masks unnecessarily.

Keep memory usage controlled.

## 10. AI Director Integration

The AI Director must be able to request semantic regions by name.

Examples:

- Brighten facial skin without brightening hair.
- Neutralize the shirt without changing the background.
- Open trouser shadows while preserving black levels.
- Adjust background brightness without affecting the person.

AI must not need to specify ellipse coordinates for these tasks.

Expose available semantic regions and diagnostic information to the AI analysis workflow.

If a region is unavailable or unreliable:

- Report the limitation.
- Avoid unsafe local corrections.
- Use a safe alternative only when justified.
- Request user input only when necessary.

Do not claim AI segmentation quality is guaranteed.

## 11. Visual Diagnostics

Generate real segmentation overlays for the processed photograph.

Required visualizations:

- Person mask.
- Background mask.
- Face or facial-skin mask.
- Hair mask.
- Upper-clothing mask.
- Lower-clothing mask.
- Combined semantic label visualization.

Also generate a comparison between:

- Existing elliptical masks.
- New semantic masks.

Overlays must use the actual DSC03037 image.

Clearly distinguish visualization colors from actual photographic color corrections.

Save diagnostics in the photograph's working directory.

Do not overwrite existing versions.

## 12. Quality Validation

Use DSC03037.ARW as the primary real-world validation photograph.

Inspect:

- Face and hair boundaries.
- Hairline.
- Shirt collar.
- Shirt sleeves.
- Shirt-to-trousers transition.
- Hands.
- Background around loose hair.
- Subject silhouette.

Verify that semantic masks improve localization compared with ellipses.

Where possible, measure:

- Mask coverage.
- Region consistency.
- Boundary behavior.
- Unintended corrections outside masks.
- Runtime.
- Peak memory usage.

Do not invent ground-truth segmentation accuracy without labeled reference masks.

Visual inspection is required in addition to automated tests.

If segmentation quality is inadequate, report the issue rather than presenting the model as production-ready.

## 13. Performance

Optimize for practical local processing.

Requirements:

- Load the model once per processing session where possible.
- Run inference once per photograph.
- Cache masks.
- Reuse masks across candidate recipes.
- Avoid repeated model initialization.
- Avoid unnecessary full-resolution intermediate arrays.
- Support sequential processing on memory-constrained machines.

Measure:

- Model loading time.
- Segmentation inference time.
- Mask refinement time.
- Peak memory consumption.
- Total additional processing overhead.

Do not introduce GPU requirements for basic operation.

## 14. Testing

Implement tests for:

- Semantic label mapping.
- Mask value ranges.
- Mask dimensions.
- Orientation handling.
- Aspect-ratio preservation.
- Letterboxing and padding.
- Mask resizing.
- Crop alignment.
- Cache validity.
- Cache invalidation.
- Atomic cache writes.
- Model failure handling.
- Missing semantic regions.
- Recipe validation.
- Mask blending.
- Preservation of unmasked pixels.
- Backward compatibility with elliptical masks.

Use deterministic fixtures for unit tests.

Separately perform real model inference on DSC03037.

Do not treat mocked inference as proof of segmentation quality.

## 15. Demonstration

Perform a real end-to-end demonstration on DSC03037.ARW.

Steps:

1. Load the actual RAW photograph.
2. Generate a suitable oriented RGB preview.
3. Run the selected local segmentation model.
4. Generate semantic masks.
5. Refine mask boundaries.
6. Cache the results.
7. Produce real mask overlays.
8. Compare semantic masks with the existing ellipse masks.
9. Create one new RAW processing version using semantic regions.
10. Generate before/after images and relevant detail crops.
11. Perform AI visual review.
12. Present the actual images and overlays in the Codex conversation using supported mechanisms.

Do not approve the photograph.

Do not export final files.

Do not process the remaining 71 photographs.

Preserve v001–v004 and all existing candidate comparisons.

## 16. Integration with Autonomous Processing

The semantic segmentation system must be compatible with the planned autonomous commercial processing workflow.

Target architecture:

**RAW → Local Segmentation → AI Artistic Analysis → Recipe Generation → Preview Exploration → Full RAW Processing → AI Review → Optional Correction → Chat Report**

Segmentation must not count as an additional full-resolution RAW rendering pass.

Maintain the limit of two full-resolution processing passes per photograph.

Reuse cached masks during optional correction.

The AI must remain responsible for artistic decisions.

The segmentation model must remain responsible for identifying image regions.

The RAW engine must remain responsible for applying pixel-level corrections.

## 17. Git and Safety

- Never modify original RAW photographs.
- Preserve existing processing versions.
- Do not overwrite prior reports or manifests.
- Do not approve photographs automatically.
- Do not export without authorization.
- Do not upload user images to external services.
- Do not commit model weights.
- Do not introduce unnecessary dependencies.
- Run relevant tests.
- Review the Git diff.
- Create a local commit after successful verification.
- Never push without explicit permission.

Respect the existing Git configuration.

Do not create a new Git repository.

## 18. Documentation

Update:

- `AGENTS.md`, where necessary.
- `docs/ROADMAP.md`.
- `docs/ai_workflow.md`.
- `docs/processing_pipeline.md`.
- Relevant quality-control documentation.
- `README.md`.

Document:

- Model installation.
- Model licensing.
- CPU/GPU configuration.
- Offline inference.
- Mask caching.
- Supported semantic regions.
- Known limitations.
- Troubleshooting.

## 19. Final Engineering Report

Provide a detailed report in Russian containing:

1. Selected segmentation model and rationale.
2. Models considered.
3. Architecture and integration.
4. Supported semantic classes.
5. Model installation and dependencies.
6. CPU/GPU execution behavior.
7. Coordinate-alignment implementation.
8. Mask refinement.
9. Cache implementation.
10. Recipe integration.
11. Actual DSC03037 mask overlays.
12. Before/after comparison.
13. Quality observations.
14. Performance measurements.
15. Automated tests.
16. Remaining limitations.
17. Files modified.
18. Git commit information.
19. Next recommended development steps.

Clearly distinguish implemented capabilities from future plans.

## 20. Execution Instructions

Begin with repository inspection and model feasibility evaluation.

Select a suitable local human-parsing model.

Implement the smallest robust segmentation subsystem that satisfies the requirements.

Do not stop after writing a plan.

Run real inference on DSC03037.ARW and inspect the generated masks.

Demonstrate an actual RAW processing version using semantic masks.

Run tests and verify that previous versions and original photographs remain unchanged.

Create a local Git commit only after successful verification.

Do not push.

Provide the final report in Russian.

**Success means that PhotoFlow can automatically identify the real boundaries of the face, hair, shirt, trousers, and background locally, then apply region-specific RAW corrections without the inaccurate elliptical spill shown in the existing DSC03037 examples.**