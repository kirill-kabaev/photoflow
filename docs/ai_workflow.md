# AI workflow: ограниченный Codex-assisted Director

Mode A — основной режим: Codex видит изображения, выбирает художественные параметры,
пишет Plan/AIReview и выполняет CLI. Python не умеет вызывать текущий разговор.
Пользователь говорит «обработай следующий кадр», «обработай съёмку», «одобри vNNN»,
не задаёт EV/WB/кривые и не редактирует JSON в обычной работе.

## Команды и состояния

```bash
photoflow direct SESSION --next
photoflow direct SESSION --photo PHOTO_ID
photoflow direct SESSION --batch
photoflow direct SESSION --resume
photoflow explore SESSION PHOTO_ID --file PLAN --name CANDIDATE
photoflow import-plan SESSION PHOTO_ID --file PLAN
photoflow import-review SESSION PHOTO_ID --file REVIEW
photoflow chat-report SESSION PHOTO_ID
photoflow presented SESSION --method tool_image_content
```

`--next` ограничивает запуск одним кадром, включая первый failed, не пропуская его
молча. `--batch` фиксирует список подходящих фото по priority/исходному порядку.
Новые файлы после старта не добавляются в активный scope автоматически.
`--photo` не позволяет начальной проверке затронуть другие кадры.

| Action | Действие Codex |
|---|---|
| analyze_and_plan | Осмотреть RAW preview, заполнить полное coverage и план |
| visual_review | Осмотреть реальные before/after, native details, рецепт/style/метрики |
| targeted_correction_plan | Интерпретировать конкретные corrections; изменить только adjustment_paths |
| present_chat | Показать реальные изображения и краткий русский отчёт; затем presented |
| recover | Восстановить прерванную операцию, сохранив использованный слот |
| requires_user_input | Показать проблему и задать один понятный вопрос |
| complete | Результаты ждут явного пользовательского approval; export не выполняется |

После presented агент продолжает batch через direct --resume без запроса
промежуточного approval. Сам presented не запускает следующий RAW. Ошибка или
blocked-фото останавливают scope; их нельзя молча исключать.

Состояние Director — `SESSION/director/current.json`, зеркало последнего состояния
запуска — `director/history/RUN_ID.json`. История решений/доставок сохраняется в run.
Queue JSON остаётся владельцем состояния фотографий, render_attempts и selected_version.

## Полнота плана и ограничение проходов

Analysis.coverage обязателен для directed-фото: exposure, white_balance, highlights,
shadows, facial_brightness, skin_tones, contrast, background, color_harmony,
hair_detail, clothing_texture, subject_separation, retouching. Для неприменимых
категорий AI честно объясняет отсутствие/невидимость, не выдумывает людей.
Портретный план содержит выбранные AI native detail regions. Source/preview SHA,
parent, style и schema проверяются до выполнения, а не только при импорте.

`max_full_passes` — 1 или 2, default 2. В queue резервируется RenderAttempt до
декодирования/рендеринга: версия, рецепт SHA, время, reserved/complete/failed.
Даже SIGTERM или ошибка не возвращают слот. Recover не сбрасывает счётчик.
Новые run и ветви не обходят лимит. Старые версии до введения счётчика сохранены
и не учитываются задним числом. Автоматического сброса бюджета нет; после исчерпания
нужно остановиться и обсудить ограничение с пользователем, не расширять max тайно.
Delivery export после отдельного user approval не является художественным проходом:
он заново воспроизводит тот же одобренный рецепт из RAW, как требует Master Plan.

## Preview exploration и кэш

Default максимум три explore (конфигурируется 1…6). Они всегда уменьшены, ≤1200 px
по длинной стороне и не больше половины размеров источника. Счётчик полных попыток
не меняется; слоты exploration сохраняются в run, включая неудачные попытки.

RawCache хранит read-only float32 `.npy`, metadata и checksum в атомарно
публикуемом каталоге. Ключ: исходный SHA, RawSettings, движок, библиотеки и decoder.
Разные WB/highlight settings требуют отдельной проявки LibRaw; одинаковые настройки
переиспользуются в previews, полном проходе и разрешённом экспорте. RAW SHA и
целостность cache проверяются даже на hit. Повреждённый cache не обходится молча.

Preview использует тот же process_pixels и color/tone semantics; радиус Gaussian
операций масштабируется относительно full-resolution пикселя. Resize перед нелинейной
обработкой не коммутирует с обработкой перед resize: preview не является точным
побайтовым предсказанием полного результата. Native texture оценивается после full pass.

## Визуальная проверка и коррекции

AIReview содержит assessment по facial_exposure, skin_tones, tonal_contrast,
background, hair_detail, clothing_texture, local_artifacts, texture, finish.
Inspected_artifacts связывает решение с SHA текущих before/after/comparison,
рецепта/style/метрик и заданных detail comparisons. Full-resolution texture inspection
нельзя заявить без настоящих detail crops. Это контракт и честный отчёт агента,
а не автоматическое доказательство, что модель действительно посмотрела файл.

Corrections при revise обязательны: problem, region, proposed_correction,
expected_improvement, side_effects, verification, adjustment_paths. Все поля
валидированы, пути ограничены поддерживаемыми RAW/global/masks/local/crop параметрами.
Вторая версия должна менять только рекомендованные параметры. Никаких shell/code tools
из рецепта и никаких неопределённых «сделать профессиональнее» вместо конкретного плана.

AI явно задаёт best_version и selection_rationale, связывает выбранный after с SHA.
Последняя версия не считается лучшей автоматически. После успешного review — остановка;
после второго неудовлетворительного — лучший сохранённый вариант и intervention.
Если review опубликован, но queue save прерван, выбор восстанавливается из проверенного AIReview.

## Человек и показ в разговоре

Обычные параметры выбирает AI. Вмешательство нужно при нерешаемой неоднозначности,
существенном остаточном дефекте, риске identity change, отсутствующей необходимой
функции, неожиданных ошибках и при финальном одобрении/экспорте.
`--resume --resolution TEXT` фиксирует реальное решение пользователя, не сбрасывает
лимит и не устраняет ошибку сам по себе. Не использовать его для автоматического обхода stop.

ChatReport содержит выбранную версию, до/после, детали, SHA, summary/improvements/limitations,
selection rationale и статус. Markdown сохраняется отдельно от исторических версий.
Картинки показываются доступным image tool; если UI их не отображает — прямые ссылки
и постоянный Markdown. `frontend_rendering_verified=false` по умолчанию: Python не
имеет права утверждать, что проверил экран VS Code. Presented — подтверждение агента
о передаче артефактов, не доказательство UI-рендера и не approval пользователя.
В реальной проверке DSC03037 пользователь подтвердил видимость изображений прямо
в разговоре. Это отдельное свидетельство текущего клиента в presentation_evidence.json,
а не свойство всех установок Codex и не изменение прежнего bundle задним числом.

## Граница автономности и Mode B

Mode A работает в активной беседе. По завершении turn/сессии Python не запускает
разговор заново: требуется продолжение агентом/пользователем. Это не unattended daemon.
Mode B API не реализован и не нужен для Mode A; --mode api даёт явную ошибку, без upload.
Будущий provider должен выдавать существующие строгие Plan/AIReview через этот протокол,
с opt-in, секретами из окружения, лимитами запросов/стоимости и bounded retries.
Семантических skin/hair моделей нет; эллипсы маркируются именно пространственными масками.
