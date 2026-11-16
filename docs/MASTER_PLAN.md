# PhotoFlow — AI-Driven Professional RAW Photography Processing System

## Role and Mission

You are a **Staff Software Engineer, AI Systems Architect, Computational Photography Engineer, and Professional Digital Retouching Specialist**.

Your task is to design and implement **PhotoFlow**, a local, AI-driven professional photography processing system.

PhotoFlow must allow a photographer to manage entire photoshoots using natural-language instructions through Codex in Visual Studio Code.

The central concept is:

**AI makes photographic and artistic decisions. Python performs precise, reproducible image-processing operations. The user reviews and approves the results.**

PhotoFlow must go beyond a conventional RAW converter with fixed presets.

The AI must be responsible for:

- Understanding the visual content of photographs.
- Evaluating technical and artistic quality.
- Choosing an appropriate commercial photographic style.
- Planning color correction.
- Planning and controlling portrait retouching.
- Identifying regions requiring local adjustments.
- Managing photo-processing priorities.
- Evaluating processed results.
- Revising unsuccessful edits.
- Interpreting user feedback.
- Maintaining visual consistency across a photoshoot.

Python must provide the underlying processing engine, file management, versioning, and reproducibility.

The system must work locally without requiring a web interface.

---

# 1. Development Environment

Target environment:

- Windows 11 with WSL2 Ubuntu.
- Visual Studio Code with the WSL extension.
- Codex extension in VS Code.
- Python 3.12.
- Git for source-code version control.
- Local filesystem for photographs and processing artifacts.

The project root is expected to be:

`~/projects/photoflow`

Before making changes:

1. Inspect the existing repository.
2. Identify implemented functionality.
3. Preserve working components.
4. Review the existing `AGENTS.md`.
5. Determine whether existing code can be extended safely.
6. Avoid unnecessary rewrites.

Do not assume that a feature is implemented simply because a module or function exists.

## Strict Architectural Constraints

Do not introduce:

- Web applications.
- Browser-based interfaces.
- FastAPI or Flask servers.
- Streamlit or Gradio.
- Desktop GUI frameworks.
- Database servers.
- Mandatory cloud infrastructure.
- Mandatory paid external services.

The user must interact with PhotoFlow using:

1. Natural-language prompts in Codex.
2. Python CLI commands when necessary.
3. Ordinary folders and files.
4. Markdown reports opened in VS Code.
5. A conventional image viewer for full-resolution inspection.

SQLite is not required. Use validated JSON files and a robust filesystem-based state-management implementation.

---

# 2. Fundamental Design Principles

## 2.1 AI-First Decision Making

AI must make artistic decisions rather than merely describe photographs.

For each image, AI must determine:

- What needs correction.
- Why it needs correction.
- Which regions require editing.
- Which operations are appropriate.
- How strongly each operation should be applied.
- Whether additional processing is necessary.

AI decisions must be converted into structured, validated processing recipes.

## 2.2 Deterministic Image Processing

Python must execute supported operations according to these recipes.

The processing engine must not make undocumented artistic decisions.

Given the same:

- RAW file.
- Recipe.
- Processing-engine version.
- Required model versions.
- Configuration.

The system should produce reproducible results within documented numerical and library limitations.

## 2.3 Non-Destructive Workflow

Original RAW files are immutable.

Never modify, overwrite, move, rename, or delete RAW originals as part of processing.

Every processing iteration must create a new version:

`v001`, `v002`, `v003`, etc.

Previous versions must remain accessible.

A user must be able to compare versions and restore an earlier processing recipe.

## 2.4 Professional Quality

Prioritize:

- Accurate color reproduction.
- Natural skin tones.
- Preservation of fine texture.
- Highlight and shadow detail.
- Consistent exposure.
- Natural contrast.
- Accurate facial appearance.
- Correct output color profiles.
- Maximum practical image resolution.

Do not use generative image reconstruction as a substitute for proper RAW development.

## 2.5 Honest Capability Reporting

Do not simulate AI analysis using hardcoded descriptions or fabricated model output.

Do not claim that AI retouching was performed unless a real AI component contributed to the processing decisions or masks.

When an AI capability is unavailable, clearly report the limitation and stop or use an explicitly permitted fallback.

---

# 3. Project Architecture

Implement a maintainable Python package with clear separation of responsibilities.

Suggested structure:

```text
photoflow/
│
├── src/photoflow/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   │
│   ├── ai/
│   │   ├── director.py
│   │   ├── analyzer.py
│   │   ├── planner.py
│   │   ├── retoucher.py
│   │   ├── reviewer.py
│   │   ├── providers.py
│   │   └── schemas.py
│   │
│   ├── queue/
│   │   ├── manager.py
│   │   ├── models.py
│   │   └── storage.py
│   │
│   ├── imaging/
│   │   ├── raw_decoder.py
│   │   ├── color_pipeline.py
│   │   ├── tone_mapping.py
│   │   ├── masking.py
│   │   ├── retouching.py
│   │   ├── quality_metrics.py
│   │   └── exporter.py
│   │
│   ├── workflow/
│   │   ├── processor.py
│   │   ├── revisions.py
│   │   └── approvals.py
│   │
│   └── reporting/
│       ├── preview.py
│       ├── photo_report.py
│       └── session_report.py
│
├── sessions/
├── tests/
├── docs/
├── examples/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
└── .gitignore
```

This is a suggested structure, not a requirement to create empty placeholder modules.

Prefer working modules over unnecessary abstraction.

Clearly define interfaces between the AI components and the image-processing engine.

---

# 4. Photoshoot Directory Structure

Each photoshoot must have a separate directory.

Example:

```text
sessions/
└── Studio_Portraits_01/
    │
    ├── raw/
    │   ├── DSC03037.ARW
    │   ├── DSC03041.ARW
    │   └── DSC03042.ARW
    │
    ├── queue.json
    ├── session_style.json
    ├── review.md
    │
    ├── work/
    │   ├── DSC03037/
    │   │   ├── analysis.json
    │   │   ├── analysis.md
    │   │   │
    │   │   ├── v001/
    │   │   │   ├── before.jpg
    │   │   │   ├── after.jpg
    │   │   │   ├── before_after.jpg
    │   │   │   ├── recipe.json
    │   │   │   ├── masks/
    │   │   │   ├── report.md
    │   │   │   ├── feedback.md
    │   │   │   ├── ai_review.json
    │   │   │   └── manifest.json
    │   │   │
    │   │   └── v002/
    │   │       └── ...
    │   │
    │   └── DSC03041/
    │       └── ...
    │
    └── final/
        ├── DSC03037.jpg
        ├── DSC03037.tif
        ├── DSC03041.jpg
        └── DSC03041.tif
```

Requirements:

- `raw/` stores original photographs.
- `work/` stores analysis, intermediate versions, masks, feedback, and reports.
- `final/` stores only explicitly approved photographs.
- `review.md` provides an overview of the complete photoshoot.
- Each processing version has its own independent artifacts.

Avoid filename collisions when multiple RAW formats share the same stem. Introduce stable photo identifiers when necessary.

Large photographs, generated artifacts, and secrets must be excluded from Git.

---

# 5. AI Photo Director

Implement the AI Photo Director as the main logical coordinator.

Responsibilities:

1. Discover photographs awaiting processing.
2. Select the next photograph.
3. Obtain visual analysis.
4. Choose processing style.
5. Create or revise a processing recipe.
6. Request execution by the Python processing engine.
7. Inspect generated previews.
8. Evaluate remaining problems.
9. Decide whether another iteration is justified.
10. Submit results for user approval.

The AI Director should operate through explicit tools and validated data structures.

It must never execute arbitrary code supplied in an AI-generated recipe.

## Important: Codex vs. Autonomous API Agent

Support two distinct integration modes.

### Mode A — Codex-Assisted

This is the default development and usage mode.

Codex in VS Code acts as the AI Director and interacts with PhotoFlow using local tools and CLI commands.

The workflow is initiated by natural-language instructions in Codex.

Python exposes machine-readable context, recipes, queue state, and processing operations.

Codex must be able to:

- Inspect generated previews using its available image-inspection capabilities.
- Read photo metadata and reports.
- Write validated AI analysis files.
- Create processing recipes.
- Run image-processing commands.
- Inspect results.
- Update queue state.
- React to user feedback.

Do not pretend that a Python subprocess can automatically call the Codex conversation.

Design an explicit handoff between the Python CLI and Codex.

If the running Codex environment cannot inspect images directly, provide a supported visual-analysis integration or report that the step cannot be completed.

### Mode B — API Agent

Provide an optional autonomous mode using a configurable vision-capable AI provider.

Requirements:

- Explicit opt-in before sending images externally.
- API credentials through environment variables.
- Configurable provider and model identifiers.
- Validated structured outputs.
- Tool-call limits.
- Request and cost budgets.
- Timeouts and bounded retries.
- Error handling.
- Logging of AI decisions.
- No hardcoded credentials.
- No automatic upload of full-resolution RAW files.

The API agent may process a queue without additional interactive prompts until a user-review checkpoint.

Do not require this mode to use PhotoFlow through Codex.

---

# 6. Professional AI Photo Analysis

AI analysis must evaluate the actual image, not only EXIF metadata.

Generate visual analysis inputs from the original RAW.

These may include:

- Full-frame preview.
- Face crop.
- Skin crop.
- Background crop.
- Highlight and shadow visualizations.
- Histograms and exposure statistics.

Use the minimum necessary data for effective analysis.

## Required Analysis Categories

### Image Classification

Determine the photographic category, such as:

- Studio portrait.
- Fashion/editorial portrait.
- Beauty portrait.
- Lifestyle photography.
- Product photography.
- Landscape.
- Other.

Do not assume all photographs contain people.

### Technical Image Quality

Evaluate:

- Exposure.
- Highlight clipping.
- Shadow clipping.
- White balance.
- Color cast.
- Contrast.
- Noise.
- Sharpness.
- Motion blur.
- Potential focus problems.
- Background distractions.

Use Python for measurable properties and AI for perceptual judgments.

Clearly distinguish measurements from subjective recommendations.

### Portrait-Specific Analysis

If people are present, evaluate:

- Skin color consistency.
- Uneven skin illumination.
- Visible temporary blemishes.
- Excessive redness.
- Unwanted color contamination.
- Eye lighting.
- Hair detail.
- Clothing texture.
- Background separation.

Do not identify people or infer sensitive personal characteristics.

### Artistic Recommendations

Determine:

- Appropriate commercial style.
- Desired contrast and tonal balance.
- Skin-tone direction.
- Background treatment.
- Local retouching priorities.
- Cropping recommendations.
- Risks of overprocessing.

## Analysis Artifacts

Create:

- `analysis.json`: validated structured analysis.
- `analysis.md`: detailed human-readable analysis.

Include:

- Observations.
- Quantitative evidence when available.
- Artistic judgments.
- Proposed editing operations.
- Reasons for each operation.
- Known uncertainties.
- Areas requiring human attention.

Do not fabricate objective quality scores.

---

# 7. AI-Driven Processing Queue

AI must influence the processing order and workflow.

Python must own persistent queue state.

## Queue States

Implement:

```text
pending
analyzing
ready
processing
review
revision_requested
approved
exporting
exported
failed
```

Use explicit and validated state transitions.

## Required Queue Behavior

- Scan RAW files in stable order.
- Register new photographs.
- Preserve existing statuses when rescanning.
- Allow AI to recommend task priorities.
- Document priority changes.
- Resume after interruption.
- Avoid accidental duplicate processing.
- Protect against concurrent conflicting writes.
- Record errors without losing queue state.
- Support one-photo-at-a-time processing.
- Prevent silent skipping.

Initially, process photographs sequentially.

Do not introduce parallel RAW-processing workers until the single-worker implementation is reliable.

## AI-Based Prioritization

AI may group photographs by:

- Lighting conditions.
- Similar compositions.
- Subject or background type.
- Processing complexity.
- Visual consistency requirements.

For example, AI may recommend processing one representative portrait first, obtaining approval for its appearance, and applying a consistent style to other photographs from the same lighting setup.

Preserve the original capture/filename order and allow the user to override AI priorities.

## Queue Persistence

Store queue state using JSON.

Use atomic writes and suitable locking.

Do not corrupt the queue when processing crashes.

Include timestamps, version identifiers, failure reasons, and user-approval state.

---

# 8. Professional RAW Processing Engine

Implement a reliable image-processing pipeline.

Recommended libraries include:

- rawpy / LibRaw.
- NumPy.
- OpenCV.
- Pillow.
- tifffile.
- scikit-image where useful.

Verify library compatibility and actual functionality before choosing dependencies.

## RAW Decoding

Support RAW formats that the installed LibRaw version can actually decode.

At minimum, test Sony ARW when a suitable sample is available.

Handle:

- Camera white balance.
- Exposure normalization.
- Demosaicing.
- Orientation.
- Black levels.
- Highlight recovery limitations.
- Output bit depth.
- Color-space conversion.

Preserve high precision throughout the processing pipeline.

Prefer a 16-bit or floating-point working representation.

Do not unnecessarily reduce intermediate data to 8 bits.

## Color Management

Implement explicitly documented color management.

Pay particular attention to:

- RAW camera color conversion.
- White-balance handling.
- Working color space.
- Transfer functions.
- Linear versus gamma-encoded operations.
- Accurate conversion to sRGB.
- ICC profile embedding.
- Correct 16-bit TIFF export.

Do not assume that simply writing 16-bit pixel values produces a color-managed TIFF.

Avoid double-applying exposure, gamma, or white-balance corrections.

Document the order of operations.

## Global Image Adjustments

Implement recipe-controlled:

- Exposure adjustment.
- White-balance adjustment where supported.
- Highlight recovery or compression.
- Shadow adjustment.
- White-point adjustment.
- Black-point adjustment.
- Contrast.
- Tone curves.
- Saturation.
- Vibrance or a documented approximation.
- Color-channel adjustments.
- Sharpening.
- Noise reduction.

Do not attempt to reproduce Lightroom parameter behavior exactly unless the algorithms have actually been implemented and verified.

## Image Quality

Prevent:

- Excessive sharpening.
- Strong halos.
- Unintended color shifts.
- Skin smoothing artifacts.
- Premature clipping.
- Loss of dark-hair texture.
- Destruction of clothing detail.

Track clipping and image statistics before and after processing.

---

# 9. AI Portrait Retouching

Portrait retouching is a central feature of PhotoFlow.

It must not be reduced to global exposure and saturation changes.

Implement a hybrid approach:

**AI plans and controls retouching. Specialized segmentation models and Python operations perform local pixel-level editing.**

## 9.1 Non-Generative Retouching

This is the default mode.

Preserve:

- Facial identity.
- Facial geometry.
- Natural expression.
- Skin pores.
- Hair structure.
- Clothing details.
- Original composition unless cropping is explicitly requested.

Support the following retouching operations.

### Skin Tone Correction

- Reduce excessive localized redness.
- Correct unwanted magenta or green casts.
- Improve skin-tone consistency.
- Preserve natural blush and lip color.
- Avoid unrealistic orange or yellow skin.

### Local Exposure Correction

- Brighten insufficiently illuminated facial regions.
- Improve eye-area lighting.
- Preserve natural shadows and facial volume.
- Control localized highlights.

### Dodge & Burn

Implement local tonal correction using smooth masks.

Avoid abrupt changes and visible mask boundaries.

The AI must choose:

- Target regions.
- Correction direction.
- Correction intensity.
- Feathering requirements.

### Skin Texture Preservation

Support restrained skin smoothing where appropriate.

Avoid blanket blurring.

Texture-preserving techniques should be preferred.

### Blemish Correction

Support careful correction of small temporary imperfections.

Avoid automatically removing permanent facial features, moles, scars, freckles, or distinctive characteristics without user instruction.

### Eye and Hair Enhancement

Support subtle local operations:

- Eye-area brightness adjustment.
- Controlled sharpening.
- Hair shadow recovery.
- Hair texture enhancement.

Avoid artificial eye whitening and exaggerated local contrast.

## 9.2 Segmentation and Masks

Provide a genuine mask-generation interface.

Possible implementations include supported face-landmark and segmentation models.

Select a practical local approach based on tested model availability.

Mask categories may include:

- Face skin.
- Eyes.
- Lips.
- Hair.
- Background.
- Clothing.

The system must support:

- Floating-point masks.
- Feathered boundaries.
- Mask visualization.
- Saving reusable masks.
- Correct mapping between preview and full-resolution coordinates.

Do not pretend that a face detector provides accurate skin or hair segmentation.

If a requested mask type is unsupported, report that limitation.

## 9.3 AI-Generated Masks and Decisions

AI should be able to specify regions or request available mask tools.

Mask-generation output must be validated before application.

Save masks and sufficient metadata for reproducibility.

AI must inspect mask overlays when supported and identify possible segmentation failures.

## 9.4 Optional Generative Retouching

Provide an extensible interface for image-editing models.

However:

- Keep the feature disabled by default.
- Require explicit user authorization.
- Never silently replace a face.
- Never replace the entire image when only local retouching was requested.
- Preserve the original image and processing history.
- Clearly label generatively edited versions.
- Record the model, settings, and source image.
- Report when original detail or resolution cannot be guaranteed.

The initial production workflow must prioritize non-generative processing.

---

# 10. Processing Recipes

Every processed version must have a machine-readable recipe.

Use Pydantic or another strict validation system.

The recipe must describe:

- Source photo identifier.
- Parent version.
- Processing-engine version.
- RAW development settings.
- Global color corrections.
- Tone adjustments.
- Local masks.
- Retouching operations.
- Cropping.
- Output settings.
- AI rationale.
- User feedback incorporated into the revision.

Example conceptual structure:

```json
{
  "schema_version": 1,
  "style": "natural_commercial_portrait",
  "global_adjustments": {
    "exposure_ev": 0.45,
    "contrast": 0.1,
    "highlight_recovery": 0.2
  },
  "local_adjustments": [
    {
      "target": "face_skin",
      "operation": "exposure",
      "value_ev": 0.15
    },
    {
      "target": "nose_redness",
      "operation": "local_saturation",
      "value": -0.1
    }
  ],
  "preserve_skin_texture": true
}
```

This example is illustrative.

Define an actual versioned schema with precise units, valid ranges, mask references, and supported operations.

Reject unknown or unsafe operations.

Never execute arbitrary source code from a recipe.

---

# 11. Autonomous Processing and Revision Loop

Implement a bounded AI-driven revision process.

For each photograph:

1. Load the original RAW.
2. Generate analysis previews.
3. Run AI visual analysis.
4. Generate a processing plan.
5. Validate the recipe.
6. Process the image.
7. Generate before/after previews.
8. Compute quality metrics.
9. Request AI visual review.
10. Decide whether a revision is necessary.
11. Create a new version if justified.
12. Submit the selected version for human review.

Default maximum: three automatic iterations per photograph.

Make this limit configurable.

## AI Quality Review

Compare the original and processed versions.

Check for:

- Remaining color casts.
- Underexposed faces.
- Excessive highlights.
- Blocked shadows.
- Unnatural skin tones.
- Over-smoothed skin.
- Sharpening artifacts.
- Visible retouching boundaries.
- Unexpected image changes.
- Inconsistency with the photoshoot style.

Combine measurable quality indicators with AI visual analysis.

Do not rely exclusively on self-assigned numerical quality scores.

Stop when:

- The result is ready for user review.
- The maximum iteration count is reached.
- Required AI capabilities are unavailable.
- A processing error occurs.
- Further changes risk degrading image quality.

Never automatically approve final artistic quality on behalf of the user.

---

# 12. Mandatory Before/After Previews

Every processing version must generate:

```text
before.jpg
after.jpg
before_after.jpg
```

These files are mandatory.

## before.jpg

Must represent the original RAW developed with a documented neutral baseline.

It must not be an AI-generated reconstruction.

## after.jpg

Must represent the actual processed result.

## before_after.jpg

Must contain both images side by side.

Requirements:

- Identical crop.
- Identical orientation.
- Identical dimensions.
- Clear BEFORE and AFTER labels.
- Suitable dimensions for VS Code preview.
- Correct sRGB encoding.
- No watermark or unnecessary decoration.

Generate a fresh `before.jpg` when cropping changes.

Both images must represent the same frame region.

## Additional Detail Previews

When portrait retouching is performed, optionally generate:

- Face before/after.
- Skin detail before/after.
- Eye detail before/after.
- Hair detail before/after.
- Mask overlays.

Maintain adequate resolution for visual inspection.

Do not confuse enlarged previews with actual increased image detail.

---

# 13. Markdown Visual Reports

Markdown reports are the primary user-review interface.

No graphical application is required.

## Per-Photo Version Report

Each version must contain:

`report.md`

The report must display real images using relative Markdown links.

Required sections:

### Photo Information

- Original filename.
- Photoshoot name.
- Version number.
- Current status.
- Original dimensions.
- Processing style.

### Before and After

Display the actual source preview and processed preview in a Markdown table.

Example:

```markdown
| BEFORE | AFTER |
|---|---|
| ![](./before.jpg) | ![](./after.jpg) |

## Combined comparison

![Before and After](./before_after.jpg)
```

### AI Analysis

Describe the original photograph and identified problems.

### Processing Decisions

Explain:

- Why the selected style was chosen.
- Which global corrections were applied.
- Which local corrections were applied.
- What was deliberately preserved.

### Retouching Details

Document:

- Retouched regions.
- Mask types.
- Adjustment strengths.
- Known limitations.

### AI Quality Review

Describe:

- Improvements achieved.
- Remaining concerns.
- Whether further editing is recommended.
- Whether the photograph requires manual attention.

### Processing Metadata

Include:

- Recipe reference.
- Processing version.
- Relevant quality measurements.
- Output characteristics.

### User Feedback

Link to `feedback.md`.

Every report must be readable through the built-in VS Code Markdown Preview.

Avoid absolute links to images.

## Photoshoot-Level Report

Generate:

`sessions/<session>/review.md`

This report must contain:

- Photoshoot summary.
- Total photo count.
- Queue status counts.
- Current processing progress.
- Ordered photograph list.
- Before/after thumbnails.
- Current version per photograph.
- Links to detailed reports.
- Outstanding feedback.
- Approved photographs.
- Exported photographs.
- Failed or blocked photographs.

The overview must be regenerated when relevant processing state changes.

---

# 14. Natural-Language User Feedback

The user must be able to control editing through ordinary Codex messages.

Examples:

"Make the face slightly brighter."

"Reduce redness around the nose but preserve natural skin texture."

"The background is too blue."

"Keep the color grading from v001 but use the facial retouching from v003."

"Make the entire photoshoot more consistent."

"Apply this approved portrait style to the remaining photographs."

"Revert the previous skin smoothing."

## Feedback Files

Each version must contain:

`feedback.md`

This file is editable by the user.

Example:

```markdown
# Feedback — DSC03037 v001

Status: revision_requested

## Requested changes

- Brighten the face slightly.
- Reduce nose redness.
- Preserve natural pores.
- Keep hair unchanged.

## Additional comments

The overall color grading is good.
```

Requirements:

- Never overwrite user-written feedback during report regeneration.
- Record which feedback was applied to each revision.
- Preserve historical feedback.
- Explain any request that could not be fulfilled.
- Do not silently ignore feedback.

## Revision Workflow

When feedback is received:

1. Identify the photograph and version.
2. Read the original AI analysis.
3. Read the previous recipe.
4. Read the feedback.
5. Create a revised processing plan.
6. Change only necessary adjustments.
7. Process the original RAW using the revised recipe.
8. Create the next version.
9. Generate new previews.
10. Generate a new report.
11. Preserve the previous version.

User feedback may also be submitted directly through Codex.

Codex should persist relevant feedback in the project rather than relying only on conversation history.

---

# 15. Photoshoot-Wide Style Consistency

Each photoshoot must support a shared style definition:

`session_style.json`

Store:

- Desired photographic style.
- Exposure characteristics.
- White-balance direction.
- Contrast characteristics.
- Color palette preferences.
- Skin-retouching intensity.
- Background treatment.
- Sharpening preferences.
- Export requirements.

AI must use this as guidance while allowing individual exposure and local adjustments.

Do not blindly apply identical numerical settings to photographs taken under different lighting conditions.

The AI should be able to select an approved representative photograph and use its appearance as a visual reference.

Any AI-based transfer of a style should be documented.

---

# 16. Final Output Directory

Every photoshoot must contain:

`final/`

Only explicitly approved photographs may be exported there.

## Output Formats

Support:

### JPEG

- Full available resolution after approved cropping.
- High-quality export.
- Correct sRGB profile.
- Appropriate metadata preservation.
- No unnecessary resizing.

### TIFF

- True 16-bit-per-channel export when supported.
- Correct color interpretation.
- Embedded or correctly associated ICC profile.
- Lossless storage or lossless compression.
- Maximum available resolution.

Do not label an 8-bit image as a 16-bit master.

Validate the exported bit depth and color profile programmatically.

## Export Rules

- Final exports must come from the approved recipe.
- Export from the original RAW processing pipeline, not by enlarging a JPEG preview.
- Record the source version.
- Never export an unapproved version.
- Never silently overwrite an existing final export.
- Create a clear error or explicit replacement workflow if a final file already exists.
- Preserve an export manifest.

Example:

```text
final/
├── DSC03037.jpg
├── DSC03037.tif
├── DSC03041.jpg
└── DSC03041.tif
```

Final photographs must be easy to browse with an ordinary file manager.

Do not require users to navigate through internal working directories to find approved images.

---

# 17. Command-Line Interface

Implement a practical CLI.

The installed command should be:

`photoflow`

Required commands:

```text
photoflow scan SESSION
photoflow status SESSION
photoflow queue SESSION
photoflow next SESSION

photoflow analyze SESSION PHOTO
photoflow process SESSION PHOTO
photoflow review SESSION PHOTO
photoflow revise SESSION PHOTO

photoflow approve SESSION PHOTO --version VERSION
photoflow export SESSION PHOTO

photoflow report SESSION
photoflow run SESSION --next
```

Add machine-readable JSON output where useful for agent integration.

Document which commands:

- Work fully offline.
- Require Codex-assisted analysis.
- Require a configured API provider.
- Require user approval.

Do not make `analyze` claim successful AI analysis unless real visual analysis has occurred.

Provide a convenient command for importing a validated plan prepared by Codex, if necessary.

## Desired Prompt-Based Workflow

The user should be able to issue instructions such as:

### Process a photograph

"Analyze and professionally retouch the next pending photograph in Studio_Portraits_01. Generate a report and leave it for review."

### Process a batch

"Process all pending photographs in the current session using Natural Commercial Portrait style. Generate before/after reports. Stop before final approval."

### Revise a photograph

"DSC03037 v002 has excessive skin smoothing. Reduce retouching strength while preserving the current color grading."

### Approve an image

"Approve DSC03037 v003 and export it to final."

Codex must translate these requests into validated project operations.

---

# 18. Reliability and Safety

## Filesystem Reliability

Implement:

- Atomic metadata updates.
- Safe directory creation.
- Conflict detection.
- Stable identifiers.
- Safe path handling.
- Idempotent scanning.
- Version collision protection.
- Recoverable processing failures.
- Clear error messages.

Prevent directory traversal through user-supplied session or photograph identifiers.

## Version Integrity

Every version must have a manifest recording:

- Source RAW identity and checksum.
- Parent version.
- Recipe checksum.
- Engine version.
- Referenced masks.
- Generated artifacts.
- AI provider information when applicable.
- Processing timestamp.

Processing output must not be marked complete until required artifacts are successfully written and validated.

## AI Safety

- Validate structured AI outputs.
- Restrict executable operations to supported processing tools.
- Do not execute generated shell commands as part of a recipe.
- Do not upload photographs without explicit permission.
- Do not send secrets to models.
- Use configurable request limits.
- Keep meaningful operational logs.
- Clearly distinguish AI assessment from measured evidence.

## Portrait Integrity

Do not modify facial anatomy, identity, or permanent distinguishing features unless explicitly requested.

Natural texture preservation is a default requirement.

---

# 19. Testing and Quality Assurance

Use pytest.

Avoid depending on external AI requests for ordinary automated tests.

Use synthetic test images for deterministic unit tests.

Introduce appropriate interfaces so AI provider responses can be mocked without pretending the real AI integration was tested.

## Required Test Areas

### Filesystem and Queue

- RAW discovery.
- Stable ordering.
- Duplicate detection.
- Rescanning without lost status.
- Valid state transitions.
- Recovery from interrupted operations.
- Concurrent write protection.
- Invalid path handling.

### Image Processing

- Image dimensions.
- Orientation.
- Exposure adjustment.
- Color transformation.
- Tone mapping.
- Mask blending.
- Output data types.
- JPEG export.
- Actual 16-bit TIFF export.
- Color profile metadata.

### Versioning

- v001 creation.
- v002 creation.
- Prevention of overwrites.
- Preservation of previous recipes.
- Correct parent-version relationships.

### Reporting

- Mandatory before/after files.
- Correct comparison dimensions.
- Valid relative Markdown links.
- Complete version reports.
- Current session review report.

### Feedback

- User feedback preservation.
- Feedback incorporation into revisions.
- Historical change tracking.

### Approval and Export

- Unapproved photographs cannot be exported.
- Approved photographs can be exported.
- Existing final files are protected.
- Export manifests reference approved versions.

### AI Integration

- Strict schema validation.
- Unsupported-operation rejection.
- Error handling.
- Missing-provider behavior.
- Request-limit enforcement.
- Real provider integration tested separately when configured.

## End-to-End Acceptance Test

Create a synthetic photoshoot and demonstrate:

1. Scan photographs.
2. Register the queue.
3. Analyze an image using the selected supported mode.
4. Generate a valid recipe.
5. Process the image.
6. Generate v001.
7. Create before/after previews.
8. Generate a Markdown report.
9. Submit feedback.
10. Generate v002.
11. Approve v002.
12. Export JPEG and TIFF to final.
13. Verify original source files remain unchanged.

Distinguish mocked AI tests from real visual-AI demonstrations.

When a real RAW sample is available, run a separate integration test.

---

# 20. Documentation

Create a practical README in Russian.

Document:

- Project purpose.
- Architecture.
- Installation on WSL2 Ubuntu.
- Python environment setup.
- Dependencies.
- Creating a photoshoot.
- Adding RAW photographs.
- Running AI analysis through Codex.
- Optional API configuration.
- Processing the first image.
- Opening reports in VS Code.
- Submitting feedback.
- Reprocessing photographs.
- Approving photographs.
- Final export.
- Troubleshooting.

Also create:

- `docs/architecture.md`
- `docs/ai_workflow.md`
- `docs/processing_pipeline.md`
- `docs/quality_control.md`

Technical documentation can use English identifiers, but explanatory text should be in Russian.

Update `AGENTS.md` to describe the operational rules for Codex.

---

# 21. Implementation Strategy

Implement incrementally, but do not stop at an empty scaffold.

The first objective is a functioning vertical slice that processes a photograph from input to approved output.

## Phase 1 — Foundation

Implement:

- Package structure.
- Configuration.
- CLI.
- Session discovery.
- File queue.
- Version management.
- Basic tests.

## Phase 2 — Real RAW Processing

Implement:

- RAW decoding.
- Color management.
- High-precision processing.
- Recipe-driven adjustments.
- JPEG and TIFF export.
- Quality measurements.

Test the actual image-processing pipeline.

## Phase 3 — AI Integration

Implement:

- Structured image-analysis contracts.
- Codex-assisted workflow.
- AI processing-plan contracts.
- Validated AI recipe import.
- AI Director orchestration.
- Optional API provider interface.

Demonstrate real visual AI analysis when the environment supports it.

## Phase 4 — AI Retouching

Implement:

- Available local segmentation.
- Region masks.
- Local exposure correction.
- Skin-tone correction.
- Texture-preserving processing.
- AI-driven retouching plans.
- Mask previews.

Do not substitute an overall image filter for local retouching.

## Phase 5 — Reports and Feedback

Implement:

- Mandatory before/after previews.
- Detailed version reports.
- Photoshoot-level review.
- Feedback processing.
- Revision history.

## Phase 6 — Autonomous Quality Loop

Implement:

- AI result evaluation.
- Bounded automatic revisions.
- Style consistency management.
- Queue progression.
- Review checkpoints.

## Phase 7 — Production Hardening

Implement:

- Full regression tests.
- Reliable failure recovery.
- Performance improvements.
- Export validation.
- Documentation.
- Final workflow testing.

---

# 22. Development and Git Rules

Follow these rules throughout development:

1. Inspect existing files before changing them.
2. Prefer focused modifications.
3. Avoid unnecessary dependencies.
4. Keep implementation modular.
5. Write tests alongside functionality.
6. Run relevant tests after each completed change.
7. Do not hide failed tests.
8. Do not claim functionality that has not been verified.
9. Document technical limitations.
10. Never modify or commit original photographs.
11. Exclude large generated artifacts from Git.
12. After completing a logically independent change with passing tests and a reviewed diff, create a local Git commit.
13. Never perform Git push without explicit user permission.

Do not run expensive AI experiments or external paid APIs without explicit authorization.

---

# 23. Expected Final Demonstration

The completed implementation must support this realistic scenario.

The user creates:

```text
sessions/Studio_Portraits_01/raw/
```

And places several Sony ARW photographs there.

The user tells Codex:

"Process the next photograph. Analyze it professionally, choose a commercial portrait style, perform color correction and natural portrait retouching, evaluate the result, and prepare a before/after report."

PhotoFlow must:

1. Locate the next queued RAW.
2. Perform genuine visual AI analysis.
3. Create a documented processing recipe.
4. Decode and process the RAW.
5. Perform supported local retouching.
6. Produce real before/after previews.
7. Inspect and evaluate the processed result.
8. Create additional versions if justified.
9. Generate a detailed Markdown report.
10. Update the photoshoot overview.
11. Leave the photograph awaiting user approval.

The user opens:

`review.md`

Then opens the photograph's:

`report.md`

The report visually shows the original and processed images.

The user writes:

"The face is slightly too dark, and the nose is too red. Brighten the face, reduce redness, and preserve natural skin texture."

Codex must interpret this feedback, generate a new recipe, perform the revision, and create a new version.

The user then says:

"Approve this version and export it."

PhotoFlow exports full-resolution, correctly color-managed JPEG and 16-bit TIFF files into:

`final/`

The entire workflow must be traceable and reproducible.

---

# 24. Final Engineering Report

After implementation, provide a detailed report in Russian.

The report must include:

1. Architecture implemented.
2. Complete list of created or modified files.
3. AI responsibilities and implemented capabilities.
4. Python processing-engine responsibilities.
5. Queue state-management design.
6. RAW processing capabilities.
7. Retouching capabilities.
8. Reporting and review workflow.
9. Export capabilities.
10. Tests executed and their results.
11. Real AI integration versus mocked functionality.
12. Known limitations.
13. Performance considerations.
14. Installation and usage instructions.
15. Recommended next engineering improvements.
16. Git commit information.

Include evidence, file paths, example commands, and test output summaries.

Do not provide vague statements such as "AI retouching is implemented" without identifying which model, mask generator, or actual processing operation performs the work.

---

# 25. Execution Instructions

Start working on the project immediately.

First inspect the current repository and existing `AGENTS.md`.

Then produce a concise implementation plan and begin implementing the first complete working vertical slice.

Do not stop after planning or creating placeholder modules.

Prioritize a verified end-to-end workflow over a large collection of incomplete features.

Implement the requested phases progressively, running tests and creating local commits for completed, verified changes.

If a feature depends on unavailable external AI services or unsupported model capabilities, do not fake it. Implement a clearly defined integration point, document the limitation, and continue with independently testable components.

If the full scope cannot be completed in the current execution, leave the repository in a working, tested state and provide a precise continuation plan.

The project is successful only when the user can manage a photoshoot through Codex prompts, inspect genuine before/after reports, request retouching revisions, and receive approved final photographs without using a graphical photo-editing application.