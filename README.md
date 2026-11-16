# PhotoFlow

Локальная недеструктивная обработка RAW через Codex в VS Code и Python CLI.
AI осматривает изображение и выбирает коррекции; Python проверяет рецепт,
обрабатывает оригинал и сохраняет воспроизводимую историю. Финальный экспорт
разрешается только после явного одобрения пользователя.

Работающая основа и оптимизированный Director: очередь → Sony ARW → визуальный handoff Codex → рецепт →
локальная обработка → before/after → feedback → следующая версия → AI review →
одобрение → JPEG и настоящий TIFF 16 бит с sRGB ICC. Веб-интерфейса, сервера,
обязательного облака и платных API нет.

## Установка в WSL2 Ubuntu

Требуется Python 3.12+, Linux/WSL2 и обычная файловая система с поддержкой flock,
atomic rename и hard links. Рекомендуется хранить сессии в файловой системе WSL,
а не на сетевом диске.

```bash
cd ~/projects/photoflow
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
photoflow --help
```

Зависимости: rawpy/LibRaw, NumPy, Pillow/LittleCMS, SciPy, tifffile, Pydantic.
Проверенные версии зафиксированы в `requirements-lock.txt`:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-build-isolation --no-deps -e .
```

## Создание сессии

Создайте папку съёмки и поместите туда свои RAW обычным файловым менеджером:

```bash
mkdir -p sessions/MyShoot/raw
photoflow scan sessions/MyShoot
photoflow status sessions/MyShoot
photoflow queue sessions/MyShoot
photoflow next sessions/MyShoot
```

Программа никогда не перемещает, не переименовывает и не записывает оригиналы.
Сканирование верхнего уровня `raw/` сохраняет статусы и SHA-256; изменение или
пропажа зарегистрированного оригинала — ошибка. Набор расширений — кандидаты
для LibRaw, а не обещание поддержки каждой камеры. Реально проверен Sony ARW.
ID включает короткий хеш имени файла: одинаковые stem разных форматов не конфликтуют.
Точные дубликаты содержимого видны в queue/status/review с `duplicate_of`; next
их не выбирает. Прямой analyze по ID остаётся явным выбором пользователя.
Для команд можно использовать уникальный ASCII stem, например `DSC03037`.

Все команды возвращают JSON; `--json` принимается для явного указания формата.
SESSION — путь к существующей директории с `raw/`; симлинки и `..` запрещены.

## Повседневная работа через Codex

Напишите: **«Обработай следующий кадр в Natural Commercial Portrait и покажи до/после»**.
Codex сам осматривает RAW-превью, выбирает параметры, сравнивает недорогие previews,
делает один полный проход и, только при значимой необходимости, одну коррекцию.
Вы не должны вручную задавать exposure/WB/кривые/маски или редактировать JSON.

Внутренние команды агента:

```bash
photoflow direct sessions/MyShoot --next
# Для конкретного кадра: direct sessions/MyShoot --photo PHOTO_ID
# Только по запросу обработать съёмку: direct sessions/MyShoot --batch
photoflow direct sessions/MyShoot --resume
```

Director возвращает явный action: analyze_and_plan → visual_review →
[необязательный targeted_correction_plan] → present_chat → complete.
Codex следует этим заданиям в текущем разговоре, используя import-plan/import-review
и ограниченное explore. Python не вызывает чат самостоятельно.

Полноразмерных попыток максимум две на разрешённый цикл работы с фотографией;
перезапуск, новая ветвь и новый run их не обнуляют. Ошибочная/прерванная попытка
тоже учитывается. Все старые версии сохранены. Уменьшенные previews слотов не занимают.
Обычное успешное завершение оставляет review и не требует ручного промежуточного
одобрения: для batch Codex показывает результат и переходит к следующему кадру.
Ошибки/неоднозначность/неисправленные дефекты останавливают цикл с одним вопросом.

Chat-report содержит ссылки и контрольные суммы настоящих before/after и деталей.
Codex показывает их через доступный локальный image tool; если клиент не показывает
изображения — даёт прямые ссылки и Markdown-отчёт. Отрисовка Markdown-картинок внутри
разговора не предполагается автоматически. В текущей проверке использован ImageContent,
и пользователь подтвердил видимость изображений прямо в разговоре. Python не имеет
доступа к экрану; это подтверждение относится к текущему клиенту. См. отчёт реализации.

## Отдельные команды и ручной handoff для диагностики

Ниже сохранён совместимый Stage 1 протокол; в повседневном режиме его выполняет Codex.


Попросите Codex: «Обработай следующий кадр в sessions/MyShoot, осмотри RAW-превью,
создай обоснованный план, выполни локальную коррекцию и оставь результат на review».

Codex выполняет:

```bash
photoflow run sessions/MyShoot --next
# либо: photoflow analyze sessions/MyShoot PHOTO_ID
photoflow schema plan > /tmp/plan.schema.json
```

`analyze` только проявляет RAW, измеряет пиксели и создаёт `context.json`,
`analysis_preview.jpg`, `plan.schema.json`. Он честно возвращает
`awaiting_codex_visual_analysis`, без выдуманного анализа.

Codex должен открыть локальное превью доступным инструментом просмотра изображений,
прочитать контекст и `session_style.json`, записать Plan JSON и выполнить:

```bash
photoflow import-plan sessions/MyShoot PHOTO_ID --file /tmp/plan.json
photoflow process sessions/MyShoot PHOTO_ID
photoflow review sessions/MyShoot PHOTO_ID
```

Plan содержит `analysis` и `recipe`. Обязательные контрольные суммы берутся из
контекста; описание схем и исполняемый пример: `examples/codex_handoff.py`.
После обработки Codex осматривает реальные `before_after.jpg`, маски и, при
необходимости, полноразмерные детали. Затем записывает AIReview JSON:

```bash
photoflow schema review > /tmp/review.schema.json
photoflow import-review sessions/MyShoot PHOTO_ID --file /tmp/ai-review.json
photoflow report sessions/MyShoot
```

Встроенный Python не вызывает разговор Codex и не выполняет NLP самостоятельно.
Если Codex не умеет осматривать изображения в текущей среде, этот шаг блокируется
с явным объяснением; использовать выдуманный анализ нельзя.

Откройте `sessions/MyShoot/review.md` в VS Code и Markdown Preview (`Ctrl+Shift+V`).
Каждый отчёт содержит реальные before/after, параметры, AI-суждения, измерения,
маски, ограничения и ссылку на `feedback.md`.

## Обратная связь и версии

Напишите пожелания в `work/PHOTO_ID/v001/feedback.md` или попросите Codex сохранить
их из разговора. Затем:

```bash
photoflow revise sessions/MyShoot PHOTO_ID
```

Команда возвращает предыдущий рецепт, текст и SHA-256 feedback. Codex интерпретирует
пожелания, меняет только необходимые параметры, указывает `parent_version`,
`feedback_sha256`, объяснение `feedback_applied` и невыполнимые просьбы в
`unfulfilled_feedback`. Импорт и обработка создают v002:

```bash
photoflow import-plan sessions/MyShoot PHOTO_ID --file /tmp/revised-plan.json
photoflow process sessions/MyShoot PHOTO_ID
```

Каждая итерация обрабатывает исходный RAW заново. Предыдущие рецепты и превью
сохраняются. Исходный feedback копируется в `feedback_applied.md` новой версии.
Редактирование feedback после импорта плана обнаруживается перед обработкой.
Для возврата к старой обработке Codex берёт старый рецепт, назначает текущего
родителя и создаёт новую версию; старые каталоги не заменяются.

## Одобрение и экспорт

Только после команды пользователя «Одобри версию и экспортируй»:

```bash
photoflow approve sessions/MyShoot PHOTO_ID --version v002 --by 'Имя пользователя'
photoflow export sessions/MyShoot PHOTO_ID
```

Требуется визуальная AI-проверка выбранной версии; рекомендация `revise` блокирует
одобрение до следующей проверки. `manual_attention` остаётся решением пользователя.
Финальные файлы: `final/PHOTO_ID.jpg`, `PHOTO_ID.tif`, `PHOTO_ID.manifest.json`.
ID в имени предотвращает коллизии. Полное разрешение после crop, JPEG quality 95,
RGB TIFF uint16 без потерь, встроенный sRGB ICC в обоих форматах. Разрядность,
размеры и профили проверяются программно. Финал заново строится из RAW по
одобренному рецепту, а не из JPEG-превью. Существующие файлы никогда не заменяются.

`approve` — инструмент фиксации решения пользователя, а не механизм аутентификации.
Codex обязан дождаться явного разрешения пользователя, даже если CLI технически доступен.

## Возможности и ограничения

- Offline: scan/status/queue/next, подготовка превью, импорт, process, отчёты,
  revise, approve, export, recover и priority.
- Осмотр изображений и художественные решения: Codex-assisted Mode A.
- Глобальные коррекции: exposure, RAW WB, compression светов, тени, black/white
  points, contrast, монотонная tone curve, saturation/vibrance, RGB gains,
  ограниченное sharpening/denoise, crop.
- Локальные: мягкие AI-выбранные эллипсы, exposure/dodge & burn, уменьшение
  красного цветового избытка, saturation, sharpening и ограниченное smoothing.
  Float32-маски и overlay сохраняются. Ellipse остаётся художественным инструментом;
  Stage 2 добавляет настоящие semantic masks для объектных коррекций.
- Нет установленных моделей кожи/волос, healing blemishes и генеративной ретуши.
  Постоянные особенности лица автоматически не удаляются; геометрия не меняется.
- Автономный API Mode B пока не реализован. `.env.example` описывает статус;
  приложение не читает ключи и не загружает фотографии. Это следующий отдельный этап,
  требующий opt-in, бюджетов, таймаутов и проверки реального провайдера.
- Ограниченный цикл Mode A реализован: максимум два полных прохода, явная визуальная
  проверка и выбор лучшей версии. API/background autonomy и автоматический style transfer
  не реализованы; процесс требует активного разговора Codex.
- Сохраняются параметры декодера, black levels, WB, размеры и версии библиотек.
  Полный перенос EXIF камеры/GPS в JPEG/TIFF пока отсутствует.
- LibRaw отдаёт 16-битные данные; детали, уже отсечённые декодером, не вернуть.
  Между версиями LibRaw/NumPy и оборудованием возможны численные отличия;
  экспорт при изменении версий библиотек требует новой обработки и проверки.

Подробности: [архитектура](docs/architecture.md), [AI workflow](docs/ai_workflow.md),
[pipeline](docs/processing_pipeline.md), [QA](docs/quality_control.md),
[дорожная карта](docs/ROADMAP.md).

## Восстановление и диагностика

```bash
photoflow recover sessions/MyShoot
photoflow priority sessions/MyShoot PHOTO_ID --value 10 --reason 'Репрезентативный свет'
photoflow report sessions/MyShoot
python -m pytest -q
PHOTOFLOW_RAW_SAMPLE=sessions/MyShoot/raw/PHOTO.ARW python -m pytest -q -m raw
```

- `Session is busy`: дождитесь окончания текущей операции; lock-файл не удаляйте.
  flock освобождается ОС при завершении процесса.
- После прерывания `recover` принимает только проверенную опубликованную версию
  или экспорт с полным манифестом; незавершённые операции переходят в failed.
- `.staging-vNNN` сохраняются для диагностики; их номера повторно не используются.
- Failed без версии: снова analyze → import-plan → process. Failed с версией:
  revise → новый план. Для failed export можно повторно approve старую проверенную
  версию, но сначала вручную разобрать/архивировать собственные частичные результаты.
- После прерванного экспорта проверьте `final/` и `.export-vNNN` в work. Частичные
  финалы не перезаписываются и не удаляются автоматически при `recover`.
  Перемещать можно только созданные PhotoFlow результаты, никогда исходные RAW.
- Повреждённый JSON или checksum mismatch не обходите; разберите причину/восстановите
  собственные метаданные из резервной копии. Не меняйте контрольные суммы для сокрытия ошибки.
- Анализ шума и пор требует 100% detail-проверки: маленькое превью этого не доказывает.

## Git в текущем окружении

Штатный `.git` смонтирован пустым и только для чтения. Локальная история создана
в `.photoflow-git/`, без изменения защищённого каталога:

```bash
git --git-dir=.photoflow-git --work-tree=. log --oneline
git --git-dir=.photoflow-git --work-tree=. status --short
```

Фотографии, sessions, окружение, секреты и артефакты не входят в коммиты.
Git push не выполняется. В обычном клоне используйте стандартные команды Git.

## Независимые кандидаты и native detail crops

Чтобы сравнить несколько обработок из одного родителя и сохранить все файлы
предыдущей версии (включая feedback/report) неизменными:

```bash
photoflow revise SESSION PHOTO --from-version v001 --feedback-file /tmp/request.md
```

Команда сохраняет отдельный снимок запроса в `work/PHOTO/revision_requests/`,
возвращает путь и SHA-256. Новый рецепт должен использовать этот checksum и
`parent_version=v001`. Ветви сохраняются независимо, но каждый новый полноразмерный process расходует
общий лимит фотографии. Исторические v002/v003/v004 остаются доступными; для нового
выбора нескольких вариантов сначала используйте уменьшенные explore previews.
Process использует проверенный linear RAW cache, связанный с неизменным оригиналом.
Внешний файл после создания снимка автоматически повторно не перечитывается.
Обычный `revise` с редактированием feedback.md сохраняет прежнее поведение.

Рецепт поддерживает `detail_regions`: список `{id, bounds, description}`.
`bounds` содержит left/top/right/bottom в нормализованных координатах ориентированного
оригинала до crop; область должна целиком входить в итоговый crop.
В `details/ID/` сохраняются before.jpg, after.jpg, before_after.jpg и region.json.
Они получены из полноразмерных float32-данных без увеличения, с масштабом 1:1;
координаты, размеры и хеши входят в метаданные и манифест версии.
Проверка тона и разбор v001: [диагностика портрета](docs/portrait_rendering_diagnosis.md).


Оптимизация, границы автономности и доказательства реальной проверки:
[implemented_autonomous_processing.md](docs/implemented_autonomous_processing.md).

## Stage 2: локальные semantic masks

Поддержаны `person`, `background`, `face`, `face_skin`, `hair`, `skin`,
`upper_clothing`, `lower_clothing`; дополнительно `dress`, `eyes`, `lips`, если
соответствующие области действительно найдены. SCHP ATR и BiSeNet R18 выполняются
локально через ONNX Runtime, CPU по умолчанию. Одежда/кожа не подменяются эллипсами.

После однократной явной установки моделей Codex использует:

```bash
photoflow segment sessions/MyShoot PHOTO_ID
photoflow direct sessions/MyShoot --photo PHOTO_ID --semantic
```

Параметры и координаты масок пользователь не задаёт: AI осматривает diagnostics и
выбирает semantic regions. Кэш разделяется между preview, RAW-версиями и экспортом.
В текущей проверке создана не одобренная v007 DSC03037; другие 71 кадр не обработаны.
На native overlays остаются coarse края: это human parsing, не точный hair matting.

Новый явно запрошенный этап после исчерпанного бюджета фиксируется отдельным
`begin-cycle --by USER --reason REQUEST`, без удаления старых попыток. Автоматически
эта команда не вызывается; в каждом цикле по-прежнему максимум два full pass.
Только новый run или feedback бюджета не добавляют.

[Установка, лицензии, CPU/GPU и schema](docs/semantic_segmentation.md) ·
[инженерный отчёт Stage 2](docs/implemented_stage_2.md).

## Кожа и согласуемое кадрирование

Новые планы автоматически получают консервативную Natural Skin Retouch внутри
семантической `face_skin`: небольшое выравнивание цвета без размытия яркостной
текстуры. `natural_skin.enabled=false` отключает её; старые версии не меняются.
Spot Healing отключён: Director сначала осматривает разметку, исключает сомнительные
отметки и отдельно спрашивает Yes/No для каждого фото. No продолжает обычную обработку.
Успешное удаление создаёт новую RAW-версию и 100% face before/after в пределах бюджета.

Для согласуемого кадрирования добавьте `--smart-crop` к `direct ... --semantic`.
Codex визуально рекомендует 16:9 или 9:16, показывает оба реальных варианта и оригинал.
Без вашего выбора crop не применяется. Если голову/кисти невозможно сохранить,
вариант помечается непригодным. Финальный экспорт использует точную выбранную область
RAW и по-прежнему требует отдельного одобрения версии.

Для демонстрации без нового RAW-прохода существуют `skin-preview`, `spot-scan`,
`crop-propose`; решения сохраняют `spot-decision` и `crop-decision`.
[Результаты реализации и ограничения](docs/implemented_skin_healing_cropping.md).
