# Проверка качества и надёжности

Обычные тесты не требуют AI API, сети и настоящих фотографий:

```bash
.venv/bin/python -m pytest -q
```

Синтетический декодер подставляет float32 RGB-паттерн вместо LibRaw. Анализ и
review имеют provider=test_fixture: тестируется файловый/численный контракт,
а не утверждается, что AI увидел синтетический кадр.

Проверяются:

- стабильное сканирование, коллизии stem, сохранение статусов и history;
- запрет незаметной замены/пропажи RAW и повреждённого JSON;
- исключающая session-wide блокировка между настоящими процессами;
- traversal/symlinks, недопустимые переходы и отсутствие одобрения;
- sRGB transfer roundtrip, экспозиция, цвет/тон, float32, clipping metrics;
- float-маски, feathering, локальное воздействие и overlay;
- одинаковый crop/размер previews, ссылки Markdown и сохранение feedback;
- v001→feedback→v002→AI review→approve→JPEG/TIFF;
- настоящий uint16 TIFF, значения с точностью выше 8 бит, RGB ICC и размер;
- повреждённый рецепт, устаревший feedback, неверный AIReview checksum;
- сохранность предыдущих версий и отказ перезаписи final;
- recovery до/после публикации версии и ошибки декодирования;
- отсутствие выдуманного анализа на этапе prepare.

Отдельный интеграционный тест реального LibRaw:

```bash
PHOTOFLOW_RAW_SAMPLE=sessions/portrait_01/raw/DSC03037.ARW \
  .venv/bin/python -m pytest -q -m raw
```

Он действительно декодирует RAW, выполняет pipeline, создаёт preview, JPEG и TIFF,
проверяет размеры/dtype/профили и исходный SHA-256. По умолчанию этот тест skipped,
так как фотографии не поставляются в Git. Проверка художественного качества сюда
не входит; она выполняется визуально Codex и затем пользователем.

## Визуальный контроль

Проверить кожу, глаза, волосы, ткань и границы масок при 100%; отсутствие чрезмерного
smoothing и ореолов. Не принимать среднюю яркость, «резкость» или самоназначенный
AI-балл за доказательство качества. Highlight/shadow fractions описывают RGB после
декодирования, а не сенсорные данные. Цветовая нейтрализация без мишени субъективна.

## Ресурсы и ограничения проверки

Обработка последовательная. Float32 RGB 26 Мп занимает около 312 МБ на массив;
несколько промежуточных массивов и дополнительная baseline-проявка при custom WB
могут потребовать несколько ГБ RAM. TIFF 26 Мп без сжатия — около 156 МБ.
Не включать параллельных RAW workers. Для batch выделять место для версий и staging.
Время и пик памяти измеряются на реальном образце, но не являются универсальными.

Пока нет полной fault-injection матрицы для каждого fsync/power-loss, проверки всех
камер, доказанного качества пор после smoothing, полной EXIF-передачи и проверки
реального API-провайдера. Конкретный прогресс и результаты — в implemented_stage_1.md.

Регрессия коммерческого портрета дополнительно проверяет монотонность и clipping
тональных операторов, отсутствие двойной gamma, независимый LibRaw sRGB reference,
ветви от одной версии без изменения её файлов и native-кропы с учётом итогового crop.
Реальные результаты и ограничения: [portrait_rendering_diagnosis.md](./portrait_rendering_diagnosis.md).

## Оптимизированный цикл и визуальная доставка

`tests/test_director.py` проверяет полную цепочку с явно synthetic AI fixtures:
полноту начального плана, повторную валидацию перед выполнением, два/один full pass,
адресность коррекции, выбор более ранней версии, stopping/intervention, восстановление
публикации/review, защищённый chat bundle, source/cache integrity и batch/next scope.
Preview и full renderer сравниваются по цветовым semantics; счётчик декодирований
доказывает одно декодирование на RawSettings для analysis/explore/full.
`tests/test_tonal_regressions.py` дополнительно проверяет точное совпадение tiled и
whole-array глобальных операций на случайных данных и границах блоков, включая фильтры.

Реальный workflow DSC03037 не подменяется fixtures: новый RAW-derived результат
осмотрен Codex вместе с native-кропами, выбрана v006, actual comparison передан как
ImageContent, показ зафиксирован как acknowledgement агента. Пользователь затем явно
подтвердил: «Да, изображения видны в разговоре». Это проверка отображения именно
в текущем клиенте; результат записан отдельно в presentation_evidence.json.
Экран клиента Python недоступен: созданный до ответа bundle сохраняет
frontend_rendering_verified=false. Markdown-файл сам по себе не доказывает
inline-рендер разговора. Кэш, ledger, исторические версии и RAW проверяются отдельным audit.
Доказательства и ограничения: [итоговый отчёт](./implemented_autonomous_processing.md).

## Stage 2: human/face parsing

Обычные `tests/test_segmentation.py` используют явно обозначенный FixtureParser.
Они проверяют class mapping, ranges/dimensions, portrait/landscape letterbox,
nearest labels, continuous alpha, rotation/inverse, crop, отсутствие воздействия вне
predicted support для всех local operations, cache hit/invalidation/corruption,
непубликацию cache после сбоя, missing regions, recipe/intersection, version provenance,
native diagnostics и прежний approval gate. GPU fallback проверяется синтетическим
provider fixture; реальный GPU в этой среде отсутствует.

Отдельный настоящий neural test:

```bash
PHOTOFLOW_RAW_SAMPLE=sessions/portrait_01/raw/DSC03037.ARW \
PHOTOFLOW_SEGMENTATION_MODELS=.models .venv/bin/python -m pytest -q
```

Реальный test декодирует RAW и вызывает обе локальные ONNX-модели; обычные tests
не скачивают модели. Требуемые env отсутствуют — этот отдельный test skipped.
Инженерная проверка Stage 2 выполнена со всеми env, без пропусков.

Визуальный контроль остаётся обязательным: source input512 даёт coarse boundaries,
а native overlay подтверждает координаты применения, не нейросетевое разрешение26Мп.
Сравнение с ellipses — agreement с предсказанием, не ground-truth accuracy.
Настоящие наблюдения, ресурсы и сохранность: [implemented_stage_2.md](implemented_stage_2.md).

## Явное удаление прежних фото

Удаление допускается только по запросу пользователя. Сохраняются RAW, актуальные и
выбранные версии, последний экспорт каждого фото и метаданные родителей, необходимые
для проверки crop. Рецепты, исходные манифесты, feedback и счётчики попыток не меняются;
удалённые изображения и массивы перечисляются с SHA-256 в `cleanup_logs/`.

При удалении источника ранее одобренного crop сохраняется `pruned_artifacts.json`
рядом с его историческим манифестом: schema_version=1, photo_id, version,
user_request, source_sha256, manifest_sha256 и removed_artifacts (путь → исходный SHA).
Проверка существующего согласования требует неизменных decision/proposal и всех трёх
превью согласования, сохранённых recipe/decoder, их исходных checksum и совпадения
SHA удалённого after.jpg с визуальным анализом. Запись не делает неполную версию
пригодной для process/export, нового визуального анализа или нового crop-decision.
Отсутствие записи и повреждение оставшихся файлов продолжают блокировать проверку.
Эти границы и экспорт текущей версии после удаления старого источника проверяются
синтетическими сценариями в `tests/test_cropping.py` без обработки реальных фото.

Чтобы убрать старые каталоги целиком из `work/`, их JSON/Markdown сохраняются без
изменения байтов в `cleanup_logs/retired_versions/PHOTO/VERSION/`. Запись
`retired_version.json` фиксирует исходный каталог, запрос пользователя, photo/version,
source_sha256 и SHA всех сохранённых метаданных. Унаследованный crop сверяется только
с проверенными recipe/manifest/decoder; для существующего approved crop дополнительно
нужны прежние доказательства согласования и запись `pruned_artifacts.json`.
Обычный `validate_version`, новый анализ/согласование и экспорт удалённой версии
по-прежнему требуют настоящие файлы в `work/`. Архив метаданных их не заменяет.
Незавершённые staging и манифесты прежних экспортов сохраняются отдельно в
`cleanup_logs/retired_staging/` и `cleanup_logs/retired_exports/`; изображения туда
не копируются. Попытки и номера версий остаются в исходной очереди.
