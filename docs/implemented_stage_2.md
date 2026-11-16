# PhotoFlow Stage 2 — инженерный отчёт

Дата: 16 ноября 2026. Задание: [STAGE_2_SEMANTIC_SEGMENTATION.md](tasks/STAGE_2_SEMANTIC_SEGMENTATION.md).
Базовый коммит: `0b10eb5`. Реальная проверка: только **DSC03037.ARW**, `portrait_01`.
Создана **v007**, Natural Commercial Portrait, состояние **review**. Одобрение,
финальный экспорт, обработка остальных 71 кадра и Git push не выполнялись.

## 1. Выполненные milestones и приёмка

| Milestone | Реальный результат / критерий приёмки |
|---|---|
| Подготовка модели | Официальные веса/источники, закреплённые SHA и лицензии; проверенный CPU ONNX export и logits parity |
| Регионы, геометрия, кэш | Настоящие body/face predictions, нужные стабильные имена, letterbox/rotation/crop tests, атомарные reusable alpha |
| Интеграция | Строгие semantic recipes, ellipse intersection, Director context/review, exploration и одна новая RAW-версия |
| Визуальная проверка | Реальные overlays, ellipse comparison, native boundary/texture crops, AIReview и показ через ImageContent |
| Регрессия и сохранность | 127 прошедших тестов, отдельный SHA audit RAW/истории, просмотр diff перед локальным коммитом |

Это расширение существующего Stage 1/Director, а не переписывание проекта.
LibRaw, float32 linear sRGB, filesystem queue/flock/atomic JSON, feedback,
version manifests и approval-gated JPEG/16-bit TIFF сохранены.

## 2. Выбор модели

Выбрана связка **SCHP ATR** для body parsing и **BiSeNet ResNet18** для лица.
Модель тела отделяет волосы, лицо, рубашку, брюки и открытые конечности; отдельный
face parser нужен, чтобы не выдавать общий класс face за точную кожу с исключением
глаз/губ/бровей. R18 достаточен на проверенном кадре; более тяжёлый R34 не добавлен.

Рассмотрены SCHP LIP, SegFormer B2 clothes, FASHN B4 и общий person/background
segmentation. Последний не даёт clothing-level классов. SegFormer-кандидаты имеют
отдельную NVIDIA-лицензию; SCHP выбран по подходящим классам, доступности официальных
весов и фактически приемлемому CPU времени. Другие body-модели не запускались и
не представлены как визуально испытанные.

Полная таблица оценки, источники, лицензии, классы, установка и troubleshooting:
[semantic_segmentation.md](semantic_segmentation.md). MIT-лицензии репозиториев
отделены от условий тренировочных данных. Для CelebAMask-HQ документировано
non-commercial research agreement; полная коммерческая правовая очистка весов/данных
этим инженерным этапом не заявляется.

## 3. Установка, зависимости и provenance

Runtime: optional ONNX Runtime **1.23.2**, без PyTorch, Transformers и CUDA extension.
Подготовка: CPU PyTorch **2.9.1+cpu**, ONNX **1.20.1**, Python **3.12.3**.
Добавлены отдельные runtime/export lock-файлы. Все пакеты устанавливались в проектное
`.venv`; системные драйверы и WSL-конфигурация не менялись.

Официальные download endpoints — Google Drive автора SCHP, GitHub Releases автора
BiSeNet, pinned GitHub source. Скачивались только модели/исходники/лицензии, фотографии
не передавались. `.models/` исключён из Git; альтернативный путь — PHOTOFLOW_MODEL_DIR.
Inference никогда не скачивает веса автоматически.

| Артефакт | SHA-256 проверенного файла |
|---|---|
| SCHP ATR checkpoint | `e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a` |
| SCHP exported ONNX | `397ec7da31b8182025d1b89eecf0cd8080834e203b48656646e03090d21ea740` |
| BiSeNet R18 ONNX | `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f` |

SCHP architecture закреплена upstream commit `eb84c432cc697f494d99662a05f2335eb2f26095`.
Вместо устаревшего InPlaceABN CUDA extension используется проверенная inference
формула с `abs(weight)+eps`. Adaptive pooling экспортируется через те же точные
mean bins, включая неделимые размеры. Checkpoint загружается `weights_only=True`.

Скрипт установки действительно выполнен offline по уже скачанным файлам. Его parity:
pooling max difference **2.15×10⁻⁶**, ONNX/PyTorch logits max difference **5.15×10⁻⁵**,
label agreement на контрольном случайном входе **1.0**. Это проверка переноса graph,
не точность сегментации на фотографии и не сравнение с оригинальным CUDA runtime.
Полные models/export-verification JSON сохранены в рабочем audit.

## 4. Архитектура и semantic классы

`segmentation.py` содержит отдельно LocalParser, преобразования координат,
стабильный mapping классов, guided alpha, SemanticCache и диагностику. Workflow
передаёт renderer валидированный provider, Director передаёт Codex доступность,
coverage/warnings/overlays, а художественные решения остаются у AI.

Поддержаны person, background, face, face_skin, hair, skin, upper_clothing,
lower_clothing; дополнительно dress, eyes и lips при наличии предсказания.
Face — coarse ATR область с частью шеи. Face_skin — BiSeNet skin/nose внутри неё;
глаза, брови и губы исключены. Skin добавляет уши/шею и ATR bare arms/legs.
Кисти представлены arms, отдельные пальцы модель не предсказывает.
Dress не делится выдуманным алгоритмом на upper/lower clothing.

Availability — наличие достаточного support, не calibrated confidence. На текущем
кадре dress отсутствует; остальные перечисленные области реально найдены.
Если facial features не найдены или регион слишком мал, unsafe recipe отклоняется.
Никакого mock inference или автоматического ellipse fallback в реальной проверке нет.

## 5. Совмещение с полным RAW

У DSC03037 LibRaw metadata `orientation_flip=5`, oriented размер **4168×6240**.
Модель получает изображение после штатной LibRaw ориентации, без второго transpose.
Входной RGB preview **1069×1600**, из нейтрального linear RAW, +0.7 EV только для
устойчивого анализа. Нейтральное photographic before остаётся прежним.

Body letterbox: 512×512, content **342×512**, left padding **85**, top **0**.
Padding явно снимается; discrete labels масштабируются Nearest. Float alpha
масштабируется Bilinear и ограничивается соответствующим дискретным support.
Masks и ellipses применяются в oriented uncropped координатах до output crop.

Существенная находка реальной проверки: без нормализации наклона BiSeNet пропускал
глаза и брови. PCA coarse face дала угол **47.13°**. Автоматический crop выровнен
для inference, labels обратно повёрнуты Nearest. После этого модель нашла глаза,
брови и губы. Это геометрическая оценка roll, не landmarks и не гарантия для профиля.
Unit tests проверяют обратимость геометрии, portrait/landscape, padding и crop.

## 6. Refinement и кэш

Image-guided scalar filter использует реальную яркость preview, radius3, epsilon0.001.
Он формирует небольшой внутренний переход; alpha обрезается predicted support.
Нет широкого outward feather, генерации волос, ретуши или изменения текстуры фотографии.
На native overlays остаётся дискретность границ исходного input512.

Ключ: RAW SHA, ориентация/размеры/decoder metadata, модель/weights SHA, preprocessing,
refinement и pipeline version. Успешный cache публикуется staging → fsync → rename
под session lock; RAW/cache identity и файлы сверяются при чтении. Сбой не создаёт
ложный complete cache. Stage2 key:
`11698e01df03880ba6e6454c803e599c1261baa8155f49a7d21e6178da9049de`.

Маски хранятся float32 на preview resolution, labels — uint8. Полный alpha создаётся
по одной области, без дублирования всех 26 Мп массивов в каждой версии.
Принятый cache — **80779790 B**. В v006 сохранялось **728233856 B** full ellipse npy,
в v007 — **104033408 B** для единственного художественного ellipse; semantic alpha
общие. Cache нужен для дальнейшего воспроизведения; GC пока не добавлен.

Повторные segment/Director/explore/full используют тот же cache без model inference.
В ходе разработки были предварительные реальные запуски: исправлена коллизия имени
face.npy для labels/alpha, затем добавлена нормализация наклона на основании реальных
ошибок. Эти проверки не скрыты за mock и не были дополнительными full RAW-версиями.
Принятый рабочий цикл использует готовый cache; обычно inference нужен один раз на key.

## 7. Recipe, Director и бюджет проходов

Schema расширена необязательными semantic_masks/segmentation; старые ellipse-only
рецепты по-прежнему валидны. Binding cache_key обязателен. Неизвестные классы, IDs,
неверный intersection, отсутствие региона и повреждённый cache отклоняются.
Существующие local exposure/saturation/redness/sharpen/smooth работают через alpha.
В демонстрации sharpen, denoise и smoothing отключены.

`intersect_ellipse` умножает alpha на опциональный художественный свет. В v007 один
такой градиент сохранён для лица; все object masks получены нейросетями.
Version manifest сохраняет provenance, recipe binding и overlays. AIReview требует
фактические SHA этих артефактов, chat bundle включает изображения semantic masks.

До этого задания DSC03037 уже израсходовала две попытки: v005 failed и v006 complete.
Новый прямой запрос пользователя создать semantic RAW-версию потребовал отдельного
явно разрешённого этапа. Добавлен **begin-cycle** с author/reason/time и границей
в append-only ledger. Старые попытки не сбрасываются и не удаляются.
Director/resume/retry не могут автоматически открывать циклы. Внутри каждого цикла
по-прежнему максимум два full pass. Это явное уточнение прежней lifetime-политики
для нового пользовательского задания, а не скрытое заявление, что попыток всего две.

Stage 2 использовал **один** full pass — v007; суммарный ledger содержит **три**
попытки с момента введения учёта, включая старую прерванную v005. Segmentation и
уменьшенный exploration полноразмерных слотов не расходуют.

## 8. Реальный photographic результат

Полный comprehensive Plan содержит 13 категорий. Сохранены RAW WB, global exposure
+0.9 EV, умеренный contrast/кривая v006. Изменены только выбранные локальные операции:

| Область | Применение v007 |
|---|---|
| face ∩ художественный ellipse | +0.45 EV, мягкое направленное освещение |
| face_skin | дополнительно +0.12 EV без глаз/бровей/губ |
| eyes | +0.10 EV, без отбеливания или реконструкции |
| hair | +0.22 EV по всей предсказанной массе |
| upper_clothing | +0.10 EV, saturation −0.06 |
| lower_clothing | +0.06 EV, с сохранением тёмной опоры |

Сначала осмотрен один уменьшенный semantic exploration, затем выполнен v007 из
исходного RAW-backed linear cache. Это не фильтр старого JPEG. Результат содержит
полный 4168×6240 pipeline, before/after/comparison и восемь native detail regions.

Относительно нейтрального before лицо заметно светлее, цвета согласованнее, волосы
и ткань читаются. Относительно уже хорошего v006 различие намеренно умеренное:
средняя абсолютная разница after JPEG **0.952/255**, native face **1.974/255**.
Before v006/v007 совпадает попиксельно. Это измерения различия, не баллы качества.

Выбрана **v007**: сохранён естественный коммерческий вид с более полной объектной
локализацией. v006 остаётся достойной альтернативой. Новая версия выбрана после
реального сравнения, не автоматически по номеру; сильный художественный скачок
относительно v006 не заявляется.

## 9. Качество масок и наблюдения

Настоящие overlays и фотографические native-кропы осмотрены Codex:

- Лицо/кожа: маска следует лицу, исключает предсказанные глаза, брови и губы.
  У линии роста волос видны coarse steps; родинки/поры в фото сохранены.
- Волосы: выделена основная масса с её асимметрией, вместо нескольких пятен.
  Тонкие свободные пряди и просветы модель местами пропускает/объединяет.
- Воротник/рукава: гораздо полнее, чем центральный shirt ellipse; мелкая граница
  воротника и манжеты не pixel-perfect.
- Рубашка/брюки: правильная общая линия пояса; ременные петли и тонкая складка
  могут смешиваться. Рубашка не перекрашена в белую.
- Кисти/силуэт: обе кисти найдены, но контур пальцев и просветы огрублены.
  Сильные коррекции на этих границах без дополнительного контроля не рекомендуются.
- На фотографическом v007 при выбранных сдержанных силах заметных жёстких ореолов
  в просмотренных областях нет. Это визуальное наблюдение на одном кадре.

Coverage по предсказанным preview labels: person **42.386%**, background **57.614%**,
face **2.322%**, face_skin **1.716%**, hair **4.391%**, skin **4.156%**,
upper_clothing **16.468%**, lower_clothing **17.169%**. Наборы пересекаются,
поэтому их нельзя все суммировать. Ground-truth IoU/accuracy не выдумывались.

В сравнении старых эллипсов alpha вне новых predicted masks: face_skin **13.16%**,
hair **21.38%**, shirt **0.27%**. Это agreement с предсказанием, не доказательство
истинной ошибочности каждого пикселя ellipse. Shirt ellipse почти весь лежал внутри
рубашки, но пропускал большую часть рукавов/воротника — важное отличие покрытия.

Float metrics v007: хотя бы один канал ≥1 у **678** пикселей (**0.00261%**),
полный RGB black у **2** пикселей. Это не sensor clipping и не оценка красоты.
Clipping fraction выше v006; значимых потерянных областей в осмотренных деталях
не обнаружено. Более сильные правки без нового осмотра не оправданы.

## 10. Реальные изображения и постоянные отчёты

В разговор переданы actual ImageContent: before/after v007, masks, labels/legend,
ellipse comparison и native detail crops. Использован поддержанный view_image,
а не предположение о рендере Markdown. В этом клиенте пользователь ранее подтвердил
видимость изображений. Python по-прежнему не имеет доступа к экрану; это не approval.

- [ДО](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/before.jpg),
  [ПОСЛЕ](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/after.jpg),
  [before/after](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/before_after.jpg).
- [Применённые semantic masks](../sessions/portrait_01/work/DSC03037-9b413174c237/stage2_verification/v007_applied_masks.jpg),
  [все обязательные регионы](../sessions/portrait_01/work/DSC03037-9b413174c237/segmentation/11698e01df03880ba6e6454c803e599c1261baa8155f49a7d21e6178da9049de/overlays.jpg).
- [Combined labels с legend](../sessions/portrait_01/work/DSC03037-9b413174c237/semantic_details/v007-11698e01df03880ba6e6454c803e599c1261baa8155f49a7d21e6178da9049de/labels-with-legend.jpg),
  [native boundary report](../sessions/portrait_01/work/DSC03037-9b413174c237/semantic_details/v007-11698e01df03880ba6e6454c803e599c1261baa8155f49a7d21e6178da9049de/report.md).
- [Ellipse/semantic comparison](../sessions/portrait_01/work/DSC03037-9b413174c237/mask_comparisons/v006-11698e01df03880ba6e6454c803e599c1261baa8155f49a7d21e6178da9049de/report.md),
  [v006/v007 after](../sessions/portrait_01/work/DSC03037-9b413174c237/stage2_verification/versions_after.jpg).
- [Version report](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/report.md),
  [recipe](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/recipe.json),
  [AIReview](../sessions/portrait_01/work/DSC03037-9b413174c237/v007/ai_review.json),
  [chat report](../sessions/portrait_01/work/DSC03037-9b413174c237/chat_reports/7f24a01b2aa74d8da6f42496025467c7/report.md).

Эти ссылки доступны в текущей рабочей папке; фотографии и рабочие отчёты исключены
из Git. При клонировании одних исходников пользовательских изображений там не будет.

## 11. Производительность и CPU/GPU

| Операция | Измерение |
|---|---|
| Offline SCHP подготовка и parity, без download | 14.46 с CLI, peak RSS 1733248 KiB |
| Успешный final segmentation, тёплый RAW cache, холодная загрузка моделей | 7.30 с CLI, peak RSS 1306516 KiB |
| SCHP loading / inference | 1.457 / 0.718 с |
| BiSeNet loading / inference | 0.152 / 0.134 с, один face crop |
| Refinement phase, включая запись alpha | 1.340 с |
| Segmentation total до публикации | 6.099 с, без предварительной проверки/чтения RAW cache |
| Повторный segment cache hit | 0.97 с CLI, peak RSS 79032 KiB, модели не загружаются |
| v007 полный RAW pipeline/reports, оба RAW cache hit | 23.65 с CLI, peak RSS 1822768 KiB |

Внутренняя метрика v007 до публикации — 21.246 с. Суммарное дополнительное время
первой сегментации относительно уже имеющейся RAW-базы — указанные 7.30 с; повторная
проверка cache стоит около секунды. Cold RAW decode сюда не включён. Измерения
конкретных команд не являются универсальным benchmark или гарантией ускорения.

CPU выбран явно, доступного CUDA provider не было. Optional auto/fallback реализован
и протестирован fixtures; реального GPU измерения нет. Сессии ONNX переиспользуются
внутри процесса; новый CLI процесс не делает повторный inference при cache hit.
Обработка последовательная, model input512 и один full alpha buffer за раз.

## 12. Автоматические тесты и сохранность

Итоговый запуск после последних изменений:

```bash
PHOTOFLOW_RAW_SAMPLE=sessions/portrait_01/raw/DSC03037.ARW \
PHOTOFLOW_SEGMENTATION_MODELS=.models .venv/bin/python -m pytest -q
```

**127 passed in 45.47s**, без skipped/failures: 124 обычных offline проверки и
3 отдельных RAW/model integration tests. Реальные tests: decode/process/JPEG/TIFF
в pytest temp; независимый LibRaw sRGB reference; настоящий SCHP/BiSeNet inference
на RAW. Тестовые файлы не являются новыми версиями съёмки или final export.

Новые tests проверяют mapping, диапазоны/dimensions, ориентацию/aspect/letterbox,
resizing/rotation/crop, guided alpha, cache hit/invalidation/atomic failure,
повреждённый cache/модель, missing region, строгий recipe/intersection,
нетронутые pixels вне alpha для всех local operations, Director semantic preview,
native diagnostics, сохранность версии, approval gate и явные cycle budgets.
Все synthetic model responses обозначены test_fixture и не подменяют реальный запуск.

Дополнительно: compileall src/scripts/tests, pip check, проверка Markdown-ссылок,
git diff --check и просмотр diff. Export source/code остаётся approval-gated.

Финальный audit подтвердил:

- все **72 RAW** имеют прежние имена/SHA-256;
- **201 исторический файл** (версии v001–v004/v006, staging v005, старые сравнения
  и chat report) не изменён; inventory старых version/comparison каталогов совпадает;
- **71 прочая запись queue** семантически неизменна, все остаются pending;
- прежние render_attempts и history prefix сохранены;
- манифесты всех шести завершённых версий, segmentation cache и новый chat bundle валидны;
- DSC03037 review, selected v007, approved_version отсутствует, final/ пуст.

Доказательство: [integrity_after.json](../sessions/portrait_01/work/DSC03037-9b413174c237/stage2_verification/integrity_after.json).
Performance/test logs, models/provenance, visual inspection SHA и планы лежат рядом.

## 13. Изменённые файлы

| Файлы | Назначение |
|---|---|
| `src/photoflow/segmentation.py` (новый) | Локальные модели, геометрия, mapping, alpha, кэш, визуальная диагностика |
| `scripts/install_segmentation_models.py` (новый) | Проверенные официальные download/export/parity |
| `src/photoflow/models.py` | Semantic contracts и явно авторизованный ProcessingCycle |
| `src/photoflow/workflow.py` | Cache binding/provenance, RAW применение и gated export, append-only cycles |
| `src/photoflow/imaging.py` | Semantic provider, прежние local operations, ellipse intersection |
| `src/photoflow/director.py` | Semantic context/preview/review и cycle-aware budget |
| `src/photoflow/cli.py` | segment, direct --semantic, begin-cycle |
| `src/photoflow/reporting.py`, `chat_reporting.py` | Семантические overlays и объяснение provenance |
| `tests/test_segmentation.py` (новый) | 31 новых cases, включая реальный inference отдельно |
| `.gitignore`, `pyproject.toml` | Исключение моделей и optional runtime extra |
| `requirements-segmentation-lock.txt`, `requirements-model-export-lock.txt` | Воспроизводимые отдельные зависимости |
| `AGENTS.md`, `README.md` | Повседневный Codex workflow и ограничения |
| `docs/ROADMAP.md`, `ai_workflow.md`, `architecture.md`, `processing_pipeline.md`, `quality_control.md` | Актуальные архитектура/QA/операции |
| `docs/semantic_segmentation.md`, `implemented_stage_2.md` (новые) | Установка, модели, классы, лицензии, инженерный результат |
| `docs/tasks/STAGE_2_SEMANTIC_SEGMENTATION.md` | Предоставленная спецификация включена без изменения текста |

Рабочие фотографии, weights, generated masks и пользовательские session-артефакты
не входят в коммит. Исторические отчёты Stage 1/диагностики/Director не переписывались.

## 14. Ограничения и следующий этап

Текущие masks пригодны для показанных умеренных локальных правок. На сильном lift,
контрастном перекрашивании, thin hair, пальцах, сложном профиле и закрытиях качество
нужно оценивать отдельно. Нет pixel-perfect hair alpha, instance separation,
внешнего произвольного raster importer, healing, генеративной ретуши и гарантии
коммерческого качества. Color calibration без мишени и GPU performance не проверены.
Условия данных/весов для коммерческого продукта требуют отдельного решения.

Следующий рекомендуемый milestone — только после отдельного запроса проверить
несколько репрезентативных сложных портретов, добавить размеченные boundary ROI
для измерения ошибок и затем уточнять alpha/модельное происхождение. Acceptance:
нет смещения/flip/crop ошибок, приемлемые волосы/пальцы/воротники при выбранных
силах, повторное использование cache, ограниченный цикл и неизменность оригиналов.
Оставшиеся 71 кадр сейчас не запускались.

## 15. Git и повседневная работа

Использован существующий `.photoflow-git`; новый репозиторий не создавался.
После проверок создаётся локальный коммит с темой:
`Add local semantic parsing, reusable RAW masks and verified Stage 2 workflow`.
Точный SHA указан в итоговом ответе и доступен локально:

```bash
git --git-dir=.photoflow-git --work-tree=. log -1 --format='%h %s'
```

Для ежедневной работы достаточно попросить Codex обработать портрет профессионально
с локальными semantic masks. Агент использует установленную модель, осматривает
overlays, выбирает recipe, выполняет ограниченный RAW workflow и показывает actual
before/after. Пользователь не задаёт координаты, EV и маски вручную. Одобрение
точной версии и final export остаются отдельными явными решениями пользователя.
