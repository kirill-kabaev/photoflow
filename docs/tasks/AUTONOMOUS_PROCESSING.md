# PhotoFlow — Autonomous Commercial RAW Processing & Chat-Native Visual Review

## Role

You are a Staff AI Systems Architect, Computational Photography Engineer, Professional Portrait Retoucher, and Python Software Engineer.

Your task is to upgrade the existing PhotoFlow project into an efficient AI-directed professional photography processing system.

The primary user interface must be the **Codex conversation in Visual Studio Code**.

The user must be able to request professional photo processing through natural language without manually adjusting exposure, white balance, contrast, colors, masks, or retouching parameters.

The AI must independently analyze photographs, select appropriate corrections, execute them through the Python RAW processing engine, visually evaluate the results, and present actual BEFORE/AFTER images directly in the Codex conversation.

The target photographic style is **Natural Commercial Portrait**.

The system must prioritize photographic quality, reproducibility, computational efficiency, and minimal user intervention.

---

## 1. Existing Project and Compatibility

Before implementing changes, read:

- `AGENTS.md`
- `docs/MASTER_PLAN.md`
- `docs/ROADMAP.md`
- `docs/PROGRESS.md`, if present
- `docs/processing_pipeline.md`

Inspect the existing source code, tests, queue management, AI handoff protocol, image-processing engine, reporting system, and Git history.

Preserve all working Stage 1 functionality, including:

- RAW decoding through LibRaw.
- High-precision image processing.
- Validated processing recipes.
- Filesystem-based queues.
- Atomic metadata updates.
- Processing versions.
- Before/after previews.
- Markdown reports.
- User feedback.
- Approval-gated exports.
- JPEG and 16-bit TIFF generation.
- Original RAW integrity checks.

Do not rewrite working components unnecessarily.

Do not introduce a web application, GUI framework, database server, or mandatory external API.

---

## 2. Primary Objective

Implement the following workflow:

**RAW → AI Analysis → Complete Processing Plan → RAW Processing → AI Visual Review → Optional Correction → Chat Report → Human Approval**

The AI must make artistic decisions independently.

The user must not be required to manually specify:

- Exposure adjustments.
- White balance.
- Contrast.
- Tone curves.
- Saturation.
- Color grading.
- Local correction strengths.
- Retouching parameters.
- Processing iteration count.

AI must generate and validate these parameters automatically.

The user should interact primarily by giving high-level instructions such as:

"Process the next photograph in Natural Commercial Portrait style."

"Process the remaining photoshoot and show me the results."

"Improve this photograph professionally."

"Approve the selected version."

---

## 3. Efficient Processing Strategy

### Maximum Two Full-Resolution Passes

Limit processing to **two full-resolution RAW rendering passes per photograph**.

Pass 1:

- Complete AI-directed processing based on comprehensive visual analysis.

Pass 2:

- Optional targeted correction based on AI visual review.

Do not implement five or more full-resolution processing iterations.

Do not perform repeated minor adjustments that could have been identified before the first processing pass.

### Thorough Initial Analysis

Before the first full-resolution processing pass, AI must analyze:

- Overall exposure.
- White balance.
- Highlight and shadow distribution.
- Facial brightness.
- Skin-tone quality.
- Tonal contrast.
- Background color.
- Color harmony.
- Hair detail.
- Clothing texture.
- Subject-background separation.
- Potential retouching requirements.

The initial recipe must address all significant identified issues simultaneously.

Avoid overly conservative recipes that result in muted or unfinished photographs.

### Efficient Preview Exploration

Use reduced-resolution previews for inexpensive recipe exploration.

Where practical:

1. Decode the RAW once.
2. Reuse suitable intermediate data.
3. Generate candidate recipes.
4. Apply candidate recipes to previews.
5. Compare the candidates visually.
6. Select the most promising recipe.
7. Execute the first full-resolution processing pass.

Preview-level exploration must not count toward the two full-resolution rendering passes.

However, avoid excessive preview generation or unnecessary AI calls.

The preview renderer must use the same color-processing semantics as the full-resolution renderer.

### Quality-Driven Second Pass

After the first full-resolution pass, AI must inspect the actual processed result.

If the result is satisfactory, stop processing.

If significant correctable problems remain, perform exactly one targeted corrective pass.

If the second pass remains unsatisfactory, stop, preserve the best available version, and report the remaining limitations.

Never perform unlimited correction loops.

---

## 4. AI Artistic Responsibilities

The AI must independently determine an appropriate Natural Commercial Portrait treatment for each photograph.

Prioritize:

- Luminous but natural skin.
- Expressive tonal depth.
- Clean and harmonious colors.
- Controlled highlights.
- Detailed shadows.
- Natural facial volume.
- Clear subject separation.
- Preserved skin texture.
- Detailed hair.
- Natural clothing texture.
- Professional overall presentation.

Avoid:

- Muted or underdeveloped results.
- Excessive global brightening.
- Artificial skin smoothing.
- Orange or gray skin.
- Crushed shadows.
- Clipped highlights.
- Excessive sharpening.
- Unnatural facial changes.
- Unnecessary generative reconstruction.

The previously processed DSC03037.ARW v001 was rejected because it appeared too muted, flat, and insufficiently polished.

Use this as evidence that technical correctness alone is not sufficient.

Do not hardcode DSC03037-specific parameters into the general processing workflow.

The AI must adapt decisions to the actual photograph.

Do not assume every commercial portrait requires a white background or identical color grading.

---

## 5. AI Visual Quality Review

After the initial processing pass, AI must visually inspect:

1. Original RAW preview.
2. Processed preview.
3. Before/after comparison.
4. Relevant detail crops.
5. Applied processing recipe.
6. Intended photographic style.
7. Available quantitative quality metrics.

Evaluate:

- Whether facial exposure is appropriate.
- Whether skin tones look natural.
- Whether tonal contrast is expressive.
- Whether background color is intentional.
- Whether hair and clothing retain detail.
- Whether local corrections are visible.
- Whether sharpening or smoothing artifacts are present.
- Whether the image appears professionally finished.

### Structured Recommendations

When improvements are necessary, AI must produce specific, actionable recommendations.

Each recommendation should identify:

- Observed problem.
- Affected image region.
- Proposed correction.
- Expected visual improvement.
- Potential side effects.
- Verification criteria.

Translate these recommendations into a validated processing recipe.

Do not merely generate generic photographic advice.

Do not invent objective quality measurements.

Do not claim that full-resolution texture was inspected unless suitable detail crops were actually examined.

### Best-Version Selection

Compare the available versions.

Select the best candidate based on:

- Visual quality.
- Technical correctness.
- Preservation of details.
- Artistic consistency.
- Absence of visible artifacts.

Do not automatically assume the newest version is the best.

Preserve all versions.

---

## 6. Chat-Native Visual Reports

The primary review interface must be the **Codex conversation in VS Code**.

After processing each photograph, Codex must present a concise report in Russian containing the actual original and processed images.

### Mandatory Chat Report

For each completed photograph, display:

**Photo information**

- Original filename.
- Photoshoot name.
- Selected version.
- Processing style.
- Processing status.

**Visual comparison**

- Actual BEFORE preview.
- Actual AFTER preview.
- Preferably side by side.

**AI processing summary**

- Main problems identified.
- Corrections applied.
- Artistic decisions made.

**AI quality assessment**

- Improvements achieved.
- Remaining limitations.
- Whether further intervention is required.

**Final status**

- Ready for user review.
- Requires user input.
- Processing failed.
- Blocked by unavailable functionality.

### Real Image Requirement

The displayed images must be the actual artifacts generated by PhotoFlow.

Never use:

- Placeholder images.
- Stock photography.
- AI-generated illustrations unrelated to the processed photograph.
- Reconstructed examples presented as actual results.

### Codex Image Rendering

Investigate the available image-display capabilities of the installed Codex extension.

Use its supported native image attachment, preview, or local artifact presentation mechanism.

Do not assume that Markdown image paths automatically render inside the Codex conversation.

Verify the behavior in the actual VS Code environment.

If native inline image rendering is unavailable:

1. Clearly report this limitation.
2. Present the closest supported image-preview mechanism.
3. Provide direct links to the actual before/after artifacts.
4. Preserve the normal Markdown visual report.

Do not claim chat-native image rendering works unless it has been demonstrated.

### Persistent Reports

Keep the existing filesystem reports:

- Per-version `report.md`.
- Session-level `review.md`.
- `before.jpg`.
- `after.jpg`.
- `before_after.jpg`.
- Detail crops when available.

These reports remain the permanent processing record.

The user should not need to open them manually during routine operation if the Codex chat can display the images.

---

## 7. Exception-Based Human Involvement

The user wants **minimal involvement**.

Do not ask for manual parameter adjustments during ordinary processing.

Do not interrupt processing for routine decisions that AI can reasonably resolve.

Request user intervention only when:

1. Artistic ambiguity cannot be resolved safely.
2. Significant defects remain after the permitted correction.
3. An operation could change facial identity or important image details.
4. A required external AI service, paid API, or data transfer needs authorization.
5. Final export approval is required.
6. An unexpected error prevents safe continuation.

When intervention is required:

- Show the actual relevant images.
- Explain the problem briefly in Russian.
- Describe the available choices.
- Ask one clear question.
- Do not require the user to understand internal recipe parameters.

The user should approve artistic results, not manually control the RAW processing engine.

### Approval Rules

AI may recommend a version for approval.

AI must never approve photographs on behalf of the user.

AI must never export unapproved photographs.

Explicit user authorization is required before final export.

---

## 8. Batch Processing

Support processing multiple photographs from a photoshoot.

The AI should:

1. Read the existing queue.
2. Select the next photograph.
3. Perform comprehensive analysis.
4. Generate a processing recipe.
5. Process the photograph.
6. Inspect the result.
7. Perform at most one correction if necessary.
8. Select the best version.
9. Present the before/after result in Codex chat.
10. Continue to the next pending photograph unless intervention is required.

Do not request approval after every ordinary processing step.

Do not silently skip failed or blocked photographs.

Preserve queue states and processing history.

If the environment cannot reliably continue the Codex-assisted workflow without further interaction, report the limitation instead of pretending to have background autonomy.

---

## 9. Codex-Assisted and API-Based Modes

Maintain a clear distinction between two operating modes.

### Mode A — Codex-Assisted

Default mode.

Codex acts as the AI Director.

Python provides:

- Queue operations.
- RAW processing.
- Validated recipes.
- Image previews.
- Quality measurements.
- Version management.
- Reports.
- Export operations.

Codex performs:

- Visual analysis.
- Artistic planning.
- Recipe generation.
- Visual quality review.
- Correction decisions.
- Best-version selection.
- User-facing reporting.

Do not pretend that Python can autonomously invoke the Codex conversation without a supported integration.

### Mode B — Optional API Agent

Preserve the architecture for future autonomous processing through a vision-capable AI API.

This mode must remain optional.

Do not require external API credentials for Codex-assisted usage.

Do not send user photographs to external services without explicit permission.

Do not implement paid API calls without authorization.

---

## 10. Implementation Requirements

Inspect the current PhotoFlow architecture before making changes.

Implement the optimized processing workflow using existing modules where practical.

Required engineering improvements:

- Configurable maximum of two full-resolution processing passes.
- Efficient preview-level recipe exploration.
- Structured AI quality-review contracts.
- Validated correction recommendations.
- Best-version selection.
- Correct stopping conditions.
- Exception-based user review.
- Chat-native visual report integration where supported.
- Preservation of persistent Markdown reports.
- Queue continuation without unnecessary interruptions.

Do not introduce unnecessary frameworks or dependencies.

Do not weaken existing RAW integrity, versioning, or export safety guarantees.

---

## 11. Testing

Add or update tests for:

- Processing-pass limits.
- Initial recipe validation.
- Optional correction behavior.
- Best-version selection.
- Stopping conditions.
- Preservation of previous versions.
- Queue continuation.
- Exception-based intervention.
- Report generation.
- Correct before/after artifact references.
- Approval-gated export.
- Failure recovery.

Use deterministic AI fixtures for automated tests.

Separately demonstrate a real Codex-assisted visual processing workflow.

Do not present mocked AI tests as proof of real visual quality.

### Real Photograph Verification

Use DSC03037.ARW for the initial demonstration.

Requirements:

1. Analyze the original photograph.
2. Generate a complete Natural Commercial Portrait recipe.
3. Process the actual RAW.
4. Inspect the result visually.
5. Perform at most one full-resolution corrective pass.
6. Select the best version.
7. Generate real before/after images.
8. Display them through the supported Codex chat mechanism.
9. Provide a concise Russian report.
10. Leave the photograph unapproved.

Do not process the remaining photographs during this initial verification.

Do not overwrite existing processing versions.

---

## 12. Git and Safety Rules

- Never modify original RAW photographs.
- Preserve existing processing history.
- Do not overwrite previous versions.
- Do not approve photographs automatically.
- Do not export without explicit approval.
- Do not upload photographs externally without permission.
- Run relevant tests.
- Review the Git diff.
- Create a local commit only after successful verification.
- Do not push without explicit permission.

Respect the repository's existing Git configuration.

Do not create additional Git repositories unnecessarily.

---

## 13. Documentation

Update:

- `AGENTS.md`
- `docs/ROADMAP.md`
- `docs/ai_workflow.md`
- Relevant processing and reporting documentation.
- `README.md`

Document the intended everyday user workflow.

Example:

The user writes:

"Process the next photograph in Natural Commercial Portrait style."

Codex independently performs the processing and replies with actual BEFORE and AFTER images and a concise Russian quality report.

If the result is acceptable, the user may approve it.

If AI detects a significant unresolved problem, it requests a decision.

The user must not need to inspect JSON recipes or manually edit image-processing parameters.

---

## 14. Final Engineering Report

Provide a detailed report in Russian.

Include:

1. Implemented architectural changes.
2. AI processing workflow.
3. Maximum-pass enforcement.
4. Preview exploration strategy.
5. Visual quality-review mechanism.
6. Best-version selection.
7. Chat image-display capabilities and verification.
8. Exception-based human involvement.
9. Queue behavior.
10. Files modified.
11. Tests executed and results.
12. Actual DSC03037 processing results.
13. Known limitations.
14. Instructions for everyday use.
15. Git commit information.

Clearly distinguish implemented features from planned capabilities.

Do not claim commercial photographic quality has been objectively guaranteed.

---

## 15. Execution Instructions

Begin by inspecting the repository and identifying the smallest set of changes required.

Implement the optimized AI-directed processing workflow and chat-native reporting integration.

Prioritize working functionality over unnecessary architectural complexity.

Do not stop after creating a plan.

Run tests and demonstrate the real DSC03037 workflow.

Preserve all existing safety guarantees.

If inline image rendering is unsupported by Codex, document the limitation and implement the best available supported alternative.

After successful verification, create a local commit.

Do not push.

Provide the final report in Russian.

**The ultimate goal is simple: the user asks AI to professionally process photographs, AI performs the work independently, and the user sees the actual BEFORE/AFTER results directly in the conversation without manually controlling editing parameters.**