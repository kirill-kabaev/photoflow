# Инженерный отчёт PhotoFlow — 16.11.2026

## Результат текущего этапа

Реализован и проверен первый сквозной Codex-assisted workflow. Это рабочий
вертикальный срез, а не завершение всех 25 разделов MASTER_PLAN. Дальнейшие этапы
и критерии приёмки перечислены в [ROADMAP.md](./ROADMAP.md).

До начала работ код не выполнялся: cli.py содержал TOML, README/AGENTS/gitignore
были пусты, тестов и зависимостей обработки не было. Существующие исходные ARW
и техническая спецификация сохранены.

## Архитектура и функции

- Python-пакет `photoflow`, устанавливаемый CLI; JSON-ответы и JSON-ошибки.
- Все основные команды из §17 плюс import-plan/import-review/schema/priority/recover.
- Строгие Pydantic-модели для очереди, анализа, рецепта, AIReview, стиля, version
  и export manifests. Неизвестные параметры и операции отклоняются; кода в рецептах нет.
- Файловая очередь со всеми 11 состояниями, историей, timestamps, priority с причиной,
  контрольными суммами, отдельными current/approved version и одобрившим пользователем.
- Стабильный filename order, ID с хешем имени, идемпотентный scan, видимые exact-content
  duplicates. Дубликаты исключены из next, но доступны по прямому ID.
- Session-wide flock между процессами; atomic JSON + fsync; безопасные относительные
  пути, запрет traversal и симлинков. Single worker, без сервера и БД.
- Транзакционное создание версий через staging/rename; checksum неизменяемых файлов,
  проверка перед approve/export, recovery прерванной обработки и экспорта.
- LibRaw AHD, WB камеры/daylight/custom, camera black levels, ориентация,
  clip/blend/reconstruct светов; uint16 decode → float32 linear sRGB.
- Управляемые рецептом exposure, highlights compression, shadows, black/white points,
  contrast, tone curve, saturation/vibrance, channel gains, sharpen/denoise и crop.
- Локальные float32 эллиптические маски с feathering, `.npy` и overlays; exposure,
  redness reduction с сохранением Y, saturation, ограниченные sharpen/smooth.
- Before/after/common comparison с корректным sRGB; одинаковая ориентация и crop;
  Markdown-анализ, подробный отчёт версии и обзор съёмки с thumbnail и статусами.
- Feedback никогда не перезаписывается при report; ревизия требует checksum и
  объяснения применения. Предыдущие рецепты/версии и снимок применённого feedback сохраняются.
- Явное одобрение точной версии; экспорт с проверкой approved recipe и библиотек,
  повторной полной обработкой оригинала, JPEG и настоящим TIFF uint16 RGB с sRGB ICC.
- Проверяются размеры, разрядность, photometric interpretation и ICC; final не
  перезаписывается. Экспортный манифест фиксирует approval и хеши файлов.

Подробное разделение ответственности: [architecture.md](./architecture.md).
Алгоритмы, единицы и порядок цвета: [processing_pipeline.md](./processing_pipeline.md).

## Реальный AI и RAW, отдельно от mock-тестов

В этой сессии Codex действительно осмотрел локальные изображения:

- `sessions/portrait_01/work/DSC03037-9b413174c237/analysis_preview.jpg`;
- `.../v001/after.jpg`, `.../v001/before_after.jpg`;
- `.../v001/masks/face_light-overlay.jpg`.

По наблюдениям составлен реальный `codex-plan.json`, импортированный в Plan.
Выполнены более тёплый RAW WB, +0.7 EV общей экспозиции, мягкие тональные коррекции
и +0.15 EV в выбранной Codex области лица. Smoothing/healing/генерация отключены.
Маска — осмысленная пространственная область, а не результат модели сегментации.
Результат осмотрен; настоящий AIReview импортирован с рекомендацией human_review.

Обработан Sony `DSC03037.ARW`: LibRaw 0.22.0, rawpy 0.26.1, flip=5,
ориентированный размер **4168×6240**. Before/after previews — 935×1400,
comparison — 1870×1432. Результат v001 оставлен на review.

Реальная сессия после повторного scan: **72 кадра, 1 review, 71 pending, 0 failed**.
`final/` пуст: пользовательские фотографии не одобрены автоматически.
Контрольные суммы **72/72 RAW совпадают** с исходной инвентаризацией; файлы не
добавлялись, не удалялись и не переименовывались. Инвентаризация проверки хранится
локально в `/tmp/photoflow-original-checksums.json` и не входит в Git.

Основной пользовательский отчёт:
`sessions/portrait_01/work/DSC03037-9b413174c237/v001/report.md`.
Общий обзор: `sessions/portrait_01/review.md`.

AI не выдавал численные художественные баллы и не заявлял, что проверил поры при 100%.
Остаточная розоватость фона, отсутствие мишени и ограничения масштаба отражены в review.

## Тесты и доказательства

Первый запуск: 35 passed, 1 skipped (RAW тест без заданного образца).
После расширения regression suite: 44 offline-теста проходят; RAW проверяется отдельно.
Полный финальный запуск с реальным образцом:

```bash
PHOTOFLOW_RAW_SAMPLE=sessions/portrait_01/raw/DSC03037.ARW \
  .venv/bin/python -m pytest -q
```

Финальный результат: **45 passed in 21.33s**, без пропусков, включая настоящий RAW integration test.
Обычный запуск без образца: **44 passed, 1 skipped**; это намеренно отдельный RAW-тест.

Покрыты scan/order/duplicates, state transitions, настоящий межпроцессный flock,
atomic-write failure, checksum/path validation, exposure/color/tone, маски/overlay,
precision и sRGB, версии v001/v002, feedback preservation/staleness, Markdown links,
AI handoff без выдумывания анализа, review checksums, approval, earlier-version export,
защита final, decoder/report failures и recovery до/после публикации.

End-to-end тест использует **синтетический декодер и test_fixture AI**, но настоящий
processing engine, файловую очередь, отчёты и JPEG/TIFF exporters. Он проходит
v001 → feedback → v002 → AIReview → approve → final и проверяет неизменность источника.
Это не проверка реального vision API.

RAW integration test действительно использует LibRaw, полное разрешение, обработку
и файловые JPEG/TIFF, проверяет ориентацию/битность/ICC и исходный SHA-256.
Тестовые outputs помещаются в pytest tmp directory, не в пользовательский final.

Дополнительно: compileall src/tests/examples, pip check, CLI --help/schema/status/next,
реальные scan/import-plan/process/import-review/report, повторный scan без потери
состояний, diff --check и проверка исключения фотографий из staged files.

## Производительность

Измеренный финальный полный запуск:
45 passed за 21.33 с, wall clock 22.26 с, peak RSS 2 910 628 КиБ (~2.8 ГиБ).
Это конкретный замер в текущем WSL/Linux-окружении, не обещание скорости на другой машине.
В этот запуск включён regression повреждённой версии при recovery.

При custom WB нейтральный baseline и результат могут требовать двух декодирований.
Новый RAW worker не запускается параллельно. Маска 26 Мп float32 занимает около 104 МБ;
данная work-папка с одним кадром занимает около 101 MiB. TIFF полного размера без
сжатия — около 156 МБ. Нужен запас RAM/диска для промежуточных массивов и истории.

## Полный список созданных или изменённых файлов

Корневые файлы:

- `.gitignore` — оригиналы, результаты, secrets, venv и metadata Git исключены;
- `.env.example` — честный статус будущего API режима, без используемых ключей;
- `pyproject.toml` — пакет, entrypoint, зависимости и pytest;
- `requirements-lock.txt` — проверенные версии зависимостей;
- `README.md` — установка, команды, workflow и диагностика на русском;
- `AGENTS.md` — правила Codex, оригиналы, анализ/review, approval и Git.

Код:

- `src/photoflow/__init__.py`;
- `src/photoflow/cli.py`;
- `src/photoflow/models.py`;
- `src/photoflow/storage.py`;
- `src/photoflow/queue.py`;
- `src/photoflow/imaging.py`;
- `src/photoflow/workflow.py`;
- `src/photoflow/reporting.py`;
- `examples/codex_handoff.py` — намеренно невалидный шаблон, требующий реального осмотра.

Тесты:

- `tests/conftest.py`;
- `tests/test_queue.py`;
- `tests/test_imaging.py`;
- `tests/test_workflow.py`;
- `tests/test_raw_integration.py`.

Документация:

- `docs/ROADMAP.md`;
- `docs/architecture.md`;
- `docs/ai_workflow.md`;
- `docs/processing_pipeline.md`;
- `docs/quality_control.md`;
- `docs/PROGRESS.md`.

`docs/MASTER_PLAN.md` прочитан полностью и не изменён; он включён в начальный локальный
коммит как исходная спецификация. Сгенерированные session queue/style/review/work
находятся только в рабочей папке и исключены из Git. Оригиналы не изменены.

## Ограничения и следующий этап

1. Нет установленной/проверенной семантической модели кожи, глаз, губ, волос и одежды.
   Нельзя называть эллипсы skin segmentation. Нет автоматического blemish healing,
   сложной локальной коррекции magenta/green или доказанного preservation всех пор.
2. API Mode B не реализован; credentials/provider/model и согласие на внешнюю передачу
   не настраивались. Нет скрытых API вызовов. Отсутствуют автономные retries/cost limits
   и автоматический quality loop. Текущая работа идёт через реальный Codex handoff.
3. Нет автоматического style transfer по approved representative; session_style
   валидируется, передаётся AI и сохраняется в версии как guidance.
4. Полный перенос camera EXIF/GPS не реализован. Рабочий gamut ограничен sRGB;
   LibRaw uint16 может отсечь часть исходных данных до последующих операций.
5. Полная cross-camera/cross-platform/power-loss regression ещё впереди. Файловые
   гарантии рассчитаны на Linux/WSL2 и cooperating flock clients, не hostile writers.
6. Публикация трёх final-файлов имеет маркер завершения manifest, но не является
   единой атомарной операцией. При жёстком прерывании частичные final и staging
   сохраняются; их разбор/архивирование требует явного действия пользователя.
7. Smoothing и sharpening имеют фиксированный радиус в пикселях; визуальная
   проверка при полном разрешении обязательна. Генеративная ретушь отсутствует.

**Следующий этап — локальная семантическая сегментация и detail review**:
проверить доступную модель, лицензию и веса; сохранить версии/хеши; добавить маски
с исключением губ/постоянных особенностей; проверить overlay и full-resolution detail.
Критерии — в ROADMAP, этап 2. Затем — отдельный opt-in API Director с бюджетами.

## Запуск и Git

Пакет уже установлен editable в `.venv`. Примеры:

```bash
.venv/bin/photoflow status sessions/portrait_01
.venv/bin/photoflow report sessions/portrait_01
.venv/bin/photoflow next sessions/portrait_01
```

Для нового workflow следовать README; следующий pending — DSC03041.ARW.
Не запускать approve/export пользовательского кадра без явного запроса пользователя.

Штатный `.git` пуст/read-only, `git init` завершился ошибкой записи config.
Создан локальный Git dir `.photoflow-git`, штатный `.git` не менялся. Проверенные
изменения фиксируются локально после тестов и review diff. SHA последнего коммита:

```bash
git --git-dir=.photoflow-git --work-tree=. log -1 --format='%h %s'
git --git-dir=.photoflow-git --work-tree=. status --short
```

Фотографии и generated artifacts в индекс не добавлялись. Git push не выполнялся.
