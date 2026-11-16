# Локальная human/face segmentation — Stage 2

Рабочий adapter: **SCHP ATR + BiSeNet ResNet18**, ID
`schp-atr-bisenet-r18-v1`. Все фотографии остаются локально. Runtime использует
ONNX Runtime; PyTorch нужен только при однократной подготовке SCHP ONNX.
Первый реальный результат и измерения: [implemented_stage_2.md](implemented_stage_2.md).

## Модели и лицензии

| Кандидат | Оценка для PhotoFlow |
|---|---|
| [SCHP](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing), ATR | Выбран для body parsing: 18 классов, одежда/волосы/лицо/руки, настоящий CPU inference. Старый Python/CUDA-зависимый extractor заменён проверенным inference-only ONNX export; архитектура и веса сохранены |
| SCHP LIP | Рассмотрен набор 20 классов, включая coat/glove; отдельной кожи лица/глаз/губ тоже нет. Веса LIP здесь не запускались |
| [SegFormer B2 clothes](https://huggingface.co/mattmdjaga/segformer_b2_clothes) | Подходящие ATR18 классы, но model card ссылается на отдельную NVIDIA-лицензию. Не выбран; реальный inference этого кандидата не выполнялся |
| [FASHN parser](https://github.com/fashn-AI/fashn-human-parser) | Современный B4 с отдельными hands/torso, но более тяжёлый и также наследует NVIDIA-лицензию. Не выбран; не выдаётся за испытанный на фото |
| [BiSeNet R18 face parsing](https://github.com/yakhyo/face-parsing) | Выбран как дополнительный небольшой face parser: отдельные skin/eyes/brows/lips/nose. R34 доступен у автора, но для проверенного кадра R18 достаточен |
| Общий person/background segmentation | Недостаточно: не предсказывает рубашку, брюки и кожу отдельно |

У SCHP и репозитория BiSeNet — MIT: [SCHP license](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing/blob/eb84c432cc697f494d99662a05f2335eb2f26095/LICENSE),
[BiSeNet license](https://github.com/yakhyo/face-parsing/blob/main/LICENSE).
Копии скачиваются рядом с моделями и проверяются SHA. Веса берутся из официального
Google Drive, указанного SCHP, и GitHub Release автора BiSeNet.

Лицензии кода, весов и тренировочных данных следует различать. BiSeNet обучен на
CelebAMask-HQ; [соглашение датасета](https://github.com/switchablenorms/CelebAMask-HQ#dataset-agreement)
содержит ограничения non-commercial research. Этот этап не является подтверждением
полного набора коммерческих прав на веса/данные. Для продукта это остаётся отдельной
проверкой происхождения и условий, а не автоматически решается MIT-лицензией кода.
SegFormer-кандидаты не выбраны, в частности из-за явного ограничения §3.3
[NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).

## Воспроизводимая установка

Из корня проекта, в проектном `.venv`, без системных изменений:

```bash
.venv/bin/python -m pip install -r requirements-segmentation-lock.txt
.venv/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-model-export-lock.txt
.venv/bin/python scripts/install_segmentation_models.py --download
```

Для runtime также объявлен optional extra `.[segmentation]`. Export tools можно
установить в отдельное окружение: для дальнейшего PhotoFlow inference они не нужны.
В проверенной установке использован Python 3.12.3, torch 2.9.1+cpu, ONNX 1.20.1,
ONNX Runtime 1.23.2. Runtime и export dependencies закреплены отдельными lock-файлами.

Скрипт установки читает только фиксированные официальные HTTPS endpoints. Он не
открывает пользовательские фотографии. Полученный файл сверяется с ожидаемым SHA
до публикации; при mismatch установка останавливается. Автоматической загрузки
весов внутри `segment`, `process`, `export` и Director нет.

Кэш моделей по умолчанию `.models/`, исключённый из Git. Можно задать
`PHOTOFLOW_MODEL_DIR=/absolute/path` и передать тот же путь установщику через
`--directory`. Для offline подготовки все исходные файлы должны уже лежать там;
установщик без `--download` использует только их.

| Файл | Размер проверенного файла | SHA-256 |
|---|---:|---|
| SCHP ATR checkpoint | 267445237 B | `e9d7c91ce3b4e7133df56b599fc817b533e3439c5e8d282a59126d2fda339a2a` |
| SCHP ONNX, локальный export | 267610306 B | `397ec7da31b8182025d1b89eecf0cd8080834e203b48656646e03090d21ea740` |
| BiSeNet R18 ONNX | 53205364 B | `0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f` |

SCHP architecture закреплена commit `eb84c432cc697f494d99662a05f2335eb2f26095`, SHA
исходника `8ab50f26dc9bc085619b1d5489399ca4c025e53d3d2c874b49c7bdd225bef321`.
Export заменяет inference InPlaceABN точной формулой `abs(weight)+eps` и static
adaptive mean bins для ONNX. Он проверяет исходный PyTorch graph против замены
pooling и ONNX logits/class agreement, публикует models.json только после parity.
Checkpoint читается `weights_only=True`; никакой код из recipe не выполняется.
Generated ONNX SHA может зависеть от export toolchain; models.json сохраняет
фактический SHA, исходный checkpoint/revision и parity, они входят в cache identity.

## CPU, GPU и offline

```bash
photoflow segment SESSION PHOTO_ID
photoflow segment SESSION PHOTO_ID --device auto
photoflow direct SESSION --photo PHOTO_ID --semantic
```

Default CPU, максимум четыре intra-op threads и один inter-op. `auto` выбирает
CUDAExecutionProvider только при его наличии; при невозможности инициализации
используется CPU. Фактические и доступные providers записываются в manifest.
В реальной проверке были CPU/Azure providers, выбран только CPU; CUDA не было.
GPU fallback покрыт fixture-тестом, реальное GPU ускорение здесь не измерялось.

Для отдельной GPU установки нужен совместимый `onnxruntime-gpu` вместо CPU wheel
и уже корректная CUDA-среда. PhotoFlow не меняет драйверы, WSL или системный CUDA.
После подготовки моделей весь inference и обработка работают offline.

## Поддерживаемые регионы

| Имя PhotoFlow | Реальное происхождение |
|---|---|
| person | ATR classes 1…17, все предсказанные foreground labels |
| background | ATR class 0 |
| face | ATR 11: coarse лицо с частью шеи, без обещания точного анатомического контура |
| face_skin | BiSeNet skin=1 и nose=10, внутри ATR face; eyes/brows/lips не включаются |
| hair | ATR 2 |
| skin | BiSeNet skin/ears/nose/neck внутри ATR face плюс ATR bare arms/legs 12…15; кисти входят в arms, отдельного hand/finger класса нет |
| upper_clothing | ATR upper-clothes=4, без выдуманного включения платьев |
| lower_clothing | ATR skirt=5 и pants=6 |
| dress | ATR 7 отдельно: верх/низ платья не изобретены |
| eyes / lips | BiSeNet 4/5 и 12/13, внутри ATR face |

Availability означает наличие достаточного predicted support (default ≥16 preview
pixels), а не calibrated confidence или гарантированную точность. Отдельные глаза,
губы и face_skin недоступны, если face inference не распознал нужные features.
На кадр рассматриваются максимум четыре face components площадью ≥64 preview pixels;
ограничение отражено в execution/transforms. Нет отдельных instance masks людей.

## Координаты и refinement

LibRaw уже применяет sensor orientation. Нейросеть получает sRGB preview из того же
oriented uncropped linear buffer, без повторного EXIF transpose. Размер preview и
экспозиция входа входят в конфигурацию (default edge1600, +0.7 EV только для анализа).
Это не JPEG-редактирование итогового снимка и не дополнительный full RAW render.

Каждый вход сохраняет aspect ratio через letterbox512. Metadata содержит исходные,
resized размеры и точные padding offsets; перед возвратом labels padding удаляется.
SCHP использует BGR ImageNet normalization, BiSeNet — RGB. Исходная горизонталь
никогда не отражается. Label maps масштабируются Nearest, float alpha — Bilinear.

Face crop автоматически определяется из SCHP. PCA coarse face оценивает roll;
crop поворачивается для inference, затем labels обратно на тот же угол Nearest.
Это не landmarks. На реальном наклоне DSC03037 без нормализации модель пропускала
глаза; с обратимой нормализацией они найдены. Если eyes/lips не найдены, безопасная
skin detail mask не предлагается.

Guided filter использует яркость реального preview, default radius3 и epsilon0.001.
Alpha ограничена исходным predicted support: сглаживается внутренний переход,
не растягивается коррекция наружу. После полного resize support снова ограничивает
alpha. Это сохраняет zero-effect вне модельной области, но не исправляет ошибку
самого class prediction. Широкое feathering и обещания strand-level matting отсутствуют.

Mask применяется до output crop. Координаты detail regions и художественных ellipses
остаются ориентированными исходными долями, как в Stage 1.

## Recipe и кэш

Расширение существующей схемы, фрагмент:

```json
{
  "segmentation": {"cache_key": "ACTUAL_CACHE_SHA256"},
  "semantic_masks": [
    {"id": "skin_light", "region": "face_skin", "description": "Освещение кожи"}
  ],
  "local_adjustments": [
    {"mask": "skin_light", "operation": "exposure", "amount": 0.12,
     "rationale": "Открыть кожу, сохранив волосы и губы"}
  ]
}
```

Полный рецепт содержит остальные обязательные поля и настоящий 64-hex cache_key.
Поддерживаются прежние exposure (EV), saturation, redness (ограниченная локальная
цветокоррекция), sharpen, smooth с прежними диапазонами. В демонстрации sharpen/smooth
отключены. `intersect_ellipse` ссылается на ID обычной ellipse; alpha умножаются.
Выполнение: прежние ellipses, затем semantic_masks, каждый список по порядку.

Ключ сегментации: RAW SHA, LibRaw orientation/dimensions/metadata, модель/weights SHA,
preprocessing, refinement и pipeline version. По умолчанию baseline RAW settings
одинаковы при всех художественных WB-рецептах. Публикация cache — staging + fsync +
rename под session lock. Source SHA, cache identity и файлы сверяются при чтении.

В `work/PHOTO/segmentation/KEY/` сохраняются uint8 labels и float32 alpha на preview
resolution, manifest, timings, warnings и overlays. По одному полному alpha buffer
реконструируется перед локальной операцией. Версии не дублируют все full arrays:
хранят segmentation.json/provenance, binding и overlays. Удаление cache ломает
воспроизводимость semantic export; автоматического GC пока нет.

## Диагностика и ошибки

Segment возвращает путь к report.md/overlays.jpg. Version report и chat bundle
показывают применённые semantic masks. Python API `compare_masks` создаёт отдельное
сравнение с явно выбранными AI ellipse IDs; `boundary_diagnostics` показывает alpha
на native RAW-кропах из version detail_regions. Эти отчёты не переписывают версии.

- Missing model: установить явно; inference не инициирует сеть самостоятельно.
- Checksum mismatch: разобрать/карантинировать изменённый cache/вес; не игнорировать.
- Session busy: дождаться завершения операции, flock не удалять.
- Missing region: сообщить отсутствие; неизвестный класс/ellipse fallback не создавать.
- Ошибка inference/cache write: staging сохранён, complete manifest не публикуется.
- Диски/CPU: операции последовательные; cache/model weights занимают место, автоматической
  очистки нет. Не удалять RAW и cache, необходимые историческим рецептам.

Результаты на одном Sony RAW не доказывают точность всех лиц, профилей, закрытий,
платьев, нескольких людей и fine hair. Перед сильными изменениями нужен осмотр
actual overlays. Обычный пользователь видит их через Codex и не рисует маски вручную.
