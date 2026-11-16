"""Markdown is the review UI. Feedback files are never regenerated."""
from collections import Counter

from .models import AIReview, Analysis, Recipe
from .storage import atomic_write, read_model


def text(path, content):
    atomic_write(path, content.encode())


def bullets(lines):
    return "\n".join(f"- {line}" for line in lines) or "- Нет записей."


def analysis_markdown(analysis):
    return f"""# Визуальный анализ — {analysis.photo_id}

Источник решений: **{analysis.provider} / {analysis.model}**. Категория: {analysis.category}.
Наличие людей: {analysis.people_present}. Это визуальные суждения AI, не объективные оценки качества.

## Наблюдения

{bullets(analysis.observations)}

## Технические суждения

{bullets([f'{k}: {v}' for k, v in analysis.technical.items()])}

## Портрет

{bullets(analysis.portrait)}

## Рекомендации

{bullets(analysis.recommendations)}

## Неопределённости

{bullets(analysis.uncertainties)}

## Внимание пользователя

{bullets(analysis.human_attention)}
"""


def version_report(session, photo, version, folder=None):
    folder = folder or session.work(photo.id, version)
    recipe = read_model(folder / "recipe.json", Recipe)
    analysis = read_model(folder / "analysis.json", Analysis)
    import json
    measures = json.loads((folder / "metrics.json").read_text())
    metadata = json.loads((folder / "decoder.json").read_text())
    review_path = folder / "ai_review.json"
    review_data = json.loads(review_path.read_text())
    if review_data.get("status") == "awaiting_codex_visual_review":
        review = "Визуальная оценка результата ещё не выполнена. Требуется Codex: осмотреть before/after и импортировать review."
    else:
        assessment = AIReview.model_validate_json(review_path.read_bytes())
        review = (f"Источник: {assessment.provider} / {assessment.model}. Решение: {assessment.recommendation}.\n\n"
                  + bullets(assessment.improvements) + "\n\nОставшиеся вопросы:\n\n"
                  + bullets(assessment.concerns) + f"\n\n{assessment.rationale}")
    masks = [f"{m.id}: эллипс {m.center}, радиусы {m.radius}, растушёвка {m.feather}. {m.description}" for m in recipe.masks]
    masks += [f"{m.id}: semantic {m.region}, intersection={m.intersect_ellipse or 'нет'}. {m.description}" for m in recipe.semantic_masks]
    operations = [f"{a.mask}: {a.operation} = {a.amount}. {a.rationale}" for a in recipe.local_adjustments]
    overlays = "\n\n".join(f"![Маска {m.id}](./masks/{m.id}-overlay.jpg)" for m in [*recipe.masks,*recipe.semantic_masks])
    mask_note = (f"Семантические маски: локальная модель {recipe.segmentation.model}, cache {recipe.segmentation.cache_key}. "
                 "[Provenance](./segmentation.json). Ellipse — отдельный художественный инструмент; "
                 "семантические границы требуют визуальной проверки, не являются точным matting."
                 if recipe.segmentation else "Маски — эллиптические области, выбранные AI, а не семантическая сегментация кожи/волос.")
    details = "\n\n".join(
        f"### {r.id}: {r.description}\n\n"
        f"[До](./details/{r.id}/before.jpg) · [После](./details/{r.id}/after.jpg) · "
        f"[Координаты и размер](./details/{r.id}/region.json)\n\n"
        f"![Детали {r.id}](./details/{r.id}/before_after.jpg)" for r in recipe.detail_regions)
    report = f"""# {photo.filename} — {version}

Сессия: {session.root.name}. Текущий статус фотографии: **{photo.state}**.
Версия: **{version}**, родитель: {recipe.parent_version or 'нет'}.
Исходные ориентированные размеры: {metadata['oriented_dimensions']}.
Стиль: **{recipe.style}**. Одобренная версия: {photo.approved_version or 'нет'}.

## До и после

| BEFORE | AFTER |
|---|---|
| ![До](./before.jpg) | ![После](./after.jpg) |

![Сравнение](./before_after.jpg)

«До»: исходный RAW, WB камеры, AHD, linear sRGB 16 бит, без auto-bright,
highlight clip; тот же crop, что у результата. Превью уменьшены только для просмотра.

## AI-анализ

{analysis_markdown(analysis)}

## Решения по обработке

{recipe.rationale}

Глобальные параметры (единицы и алгоритмы: docs/processing_pipeline.md):

```json
{recipe.global_adjustments.model_dump_json(indent=2)}
```

RAW: `{recipe.raw.model_dump_json()}`. Crop: `{recipe.crop.model_dump_json()}`.
Точное одобренное кадрирование: `{recipe.approved_crop.model_dump_json() if recipe.approved_crop else 'нет'}`.

## Локальная ретушь и маски

{bullets(masks)}

{bullets(operations)}

{overlays}

{mask_note}
Сглаживание, если запрошено: ограниченная фильтрация с весами по перепаду яркости;
сохранение пор требует визуальной проверки. Геометрия лица и композиция не меняются,
кроме явно заданного crop. Автоматическое удаление родинок/шрамов отсутствует.

## Визуальная AI-проверка

{review}

## Детальные кропы без увеличения

Кропы получены из полноразмерных float32-данных до JPEG-кодирования, масштаб 1:1.
На экране Markdown они могут отображаться уменьшенными; откройте отдельное «После» для 100%.

{details or 'Детальные области не заданы в рецепте.'}

Natural Skin Retouch: `{recipe.natural_skin.model_dump_json() if recipe.natural_skin else 'legacy/off'}`.
{measures.get('natural_skin', {})}
Выравнивается только медленное изменение цвета кожи, без размытия яркостной текстуры.

{('Spot Healing: отдельный явный Yes; те же тональные настройки, native 100% до/после удаления.\n\n![Spot Healing](./healing/before_after.jpg)\n\n[Согласие](./healing/decision.json) · [Операция](./healing/operation.json)') if recipe.spot_healing else 'Spot Healing отключён; удаления точек нет.'}

## Измерения Python

```json
{json.dumps(measures, ensure_ascii=False, indent=2)}
```

Это измерения пикселей, не художественные баллы. Сжатие светов не восстанавливает
детали, уже потерянные декодером или сенсором.

## Метаданные и воспроизводимость

Движок {recipe.engine_version}. [Рецепт](./recipe.json), [манифест](./manifest.json),
[декодер и версии библиотек](./decoder.json), [анализ](./analysis.json),
[AI-проверка](./ai_review.json). Экспорт: sRGB JPEG 8 бит и RGB TIFF 16 бит/канал с ICC.
Экспорт заново обрабатывает RAW; превью не является мастер-файлом.

## Обратная связь

[Редактировать feedback.md](./feedback.md).

Применённая обратная связь:

{recipe.feedback_applied or 'Это первая версия; обратной связи нет.'}

Невыполненные пожелания:

{bullets(recipe.unfulfilled_feedback)}
"""
    text(folder / "report.md", report)


def session_report(session, queue):
    counts = dict(sorted(Counter(p.state for p in queue.photos).items()))
    rows = []
    for photo in queue.photos:
        version = photo.selected_version or photo.current_version
        link = f"work/{photo.id}/{version}" if version else None
        filename = photo.filename.replace("|", "\\|").replace("\n", " ")
        previews = f"![До]({link}/before.jpg) ![После]({link}/after.jpg)" if link else "—"
        details = f"[{version}]({link}/report.md) · [feedback]({link}/feedback.md)" if link else "—"
        if photo.selected_version:
            details += f" · AI выбрал {photo.selected_version}; последняя {photo.current_version}"
        note = photo.error or (f"Дубликат {photo.duplicate_of}; next не выбирает, прямой analyze допустим" if photo.duplicate_of else "—")
        errors = note.replace("|", "\\|").replace("\n", " ")
        rows.append(f"| {filename} (`{photo.id}`) | {photo.state} | {photo.priority} | {details} | {photo.approved_version or '—'} | {errors} | {previews} |")
    text(session.path("review.md"), f"""# Обзор съёмки — {session.root.name}

Всего: {len(queue.photos)}. Состояния: {counts}. Обновлено: {queue.updated_at}.
Исходный порядок сохранён; приоритет влияет на выбор следующего кадра.
Только явное одобрение пользователя разрешает экспорт в [final](./final/).
`failed` требует разбора ошибки и явного retry/recover; такие кадры не скрываются.
Статус `revision_requested` означает ожидающую применения обратную связь.

| Фотография / ID | Статус | Приоритет | Версия и обратная связь | Одобрено | Ошибка | До / после |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}
""")
