# MyMicroLLM

Учебная языковая модель примерно на 250 миллионов параметров: Python,
NumPy, **ручной forward и backpropagation**. Цель проекта — разобраться,
как текст превращается в предсказание следующего токена и как производные
изменяют веса модели.

NumPy хранит матрицы `float32` и выполняет умножение через BLAS.
Циклы Python над `array.array` для такого количества весов слишком медленны.
Attention, RMSNorm, SwiGLU, cross-entropy, их производные и AdamW написаны
в исходниках. PyTorch, TensorFlow, autograd и готовые модели не используются.
Ruff, mypy и pytest нужны только для разработки. Экспериментальный CUDA
backend на Taichi сохраняет ручные производные и описан ниже.

## Установка и первый запуск

Нужны Git и [uv](https://docs.astral.sh/uv/). Команды выполняются из корня
репозитория; они подходят для PowerShell. `uv` выберет Python 3.12 по
`.python-version`, создаст `.venv` и установит зависимости из `uv.lock`.
Имя Python-пакета и команды — `microllm`.

```powershell
git clone https://github.com/Mastermind-U/mymicrollm.git
cd mymicrollm
uv sync --locked
uv run microllm info
```

Датасеты и обученные веса в Git не включены. Для первого эксперимента
скачайте небольшой Shakespeare и обучите конфигурацию на **30 976 параметров**:
2 слоя, ширина 32, 4 головы, FFN 64, контекст 64 байтовых токена.

```powershell
uv run microllm download tinyshakespeare
uv run microllm prepare tinyshakespeare
uv run microllm --threads 4 train --config configs/tiny.json --data-dir data/tokenized/tinyshakespeare --output-dir runs/demo --steps 500 --batch-size 8 --sequence-length 64 --learning-rate 0.003 --min-learning-rate 0.0003 --warmup-steps 25 --eval-every 50 --eval-batches 4 --save-every 100
```

После завершения появятся `runs/demo/checkpoint.npz` и `metrics.jsonl`.
Это маленькая учебная модель; 500 шагов не обеспечивают связный текст.

Начните изучение с [математического разбора](docs/math.md), затем читайте
[model.py](src/microllm/model.py): embedding → RMSNorm → causal attention →
SwiGLU → logits → cross-entropy → обратный проход. Обозначения формул
совпадают с комментариями; [тесты модели](tests/test_model.py) сравнивают
ручные производные с конечными разностями.

## Инференс

Генерация требует сохранённый checkpoint. Сначала выполните обучение выше
или перенесите checkpoint, созданный этим проектом, в указанный путь.
Для маленькой модели:

```powershell
uv run microllm generate --checkpoint runs/demo/checkpoint.npz --prompt "ROMEO:" --max-new-tokens 120
```

Для модели 250М после обучения из следующего раздела:

```powershell
uv run microllm generate --checkpoint runs/250m/checkpoint.npz --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

Архитектура и веса читаются из checkpoint автоматически; датасет для
инференса не нужен. `--max-new-tokens` ограничивает число новых **байтовых
токенов**, не слов. `--temperature 0` включает argmax; положительная
температура и `--top-k` задают случайную выборку, `--seed` — её повторяемость.

CLI по умолчанию выбирает окно не длиннее обучающей последовательности,
сохранённой в checkpoint. `--context-length` позволяет задать его явно
в пределах архитектуры. Подробности: `uv run microllm generate --help`.

## Обучение 250М

`configs/250m.json`: 20 слоёв, ширина 1024, 16 голов, FFN 2688,
максимальный контекст 1024 байтовых токена, **250 392 576 параметров**.
Команда `info` считает веса без создания матриц.

```powershell
uv run microllm info --config configs/250m.json
uv run microllm download tinystories
uv run microllm prepare tinystories
uv run microllm --threads 4 train --config configs/250m.json --data-dir data/tokenized/tinystories --output-dir runs/250m --steps 10000 --batch-size 1 --sequence-length 128 --accumulation-steps 4 --eval-every 100 --save-every 1000
```

Это пример запуска эксперимента, а не обещание качества после 10 000 шагов.
Работает CPU/BLAS NumPy; полное предобучение на миллиардах байтов требует
много времени. `--threads` задаётся **перед** именем команды. Больше потоков
не всегда означает большую скорость.

Веса занимают **0.933 GiB**, веса + градиенты + два момента Adam — минимум
**3.731 GiB**. Активации, временные массивы и accumulation требуют больше RAM.
Checkpoint с моментами занимает около **3 ГБ**; при сохранении нужно место
и под предыдущий файл, и под временный новый. Исходный и подготовленный
TinyStories вместе занимают около **6 ГБ** на диске.

Обучение с `--sequence-length 128` обновляет learned positions 0…127.
Для использования всех позиций нужен соответствующий этап обучения с
`--sequence-length 1024`; память attention растёт квадратично с длиной.
Обучение выбирает случайные окна с возвращением, полного прохода по эпохе нет.
KV-cache, mixed precision и распределённое обучение не реализованы.

### Экспериментальное обучение на NVIDIA GPU

Taichi выполняет прямой и ручной обратный проход на CUDA. Для Windows
с Python 3.12 и совместимой видеокартой установите дополнительную зависимость:

```powershell
uv sync --locked --extra gpu
uv run --extra gpu microllm train --device cuda --config configs/250m.json --data-dir data/tokenized/tinystories --output-dir runs/250m-gpu --steps 10000 --batch-size 1 --sequence-length 128 --accumulation-steps 4 --eval-every 100 --save-every 1000
```

`--device cuda` требует работающий CUDA backend и не переключается молча на
CPU. GPU-реализация экспериментальная, использует float32 и простые ядра
матричного умножения; её скорость и расход видеопамяти зависят от устройства.
Данные и формат checkpoint совместимы с CPU. Продолжение из checkpoint:

```powershell
uv run --extra gpu microllm train --device cuda --resume runs/250m-gpu/checkpoint.npz --data-dir data/tokenized/tinystories --output-dir runs/250m-gpu
```

Параметр `--parallel-accumulation` объединяет микробатчи одного шага в один
пакет для GPU. Его можно добавить и при `--resume`: настройки обучения и
порядок выборки данных сохраняются. Режим требует больше видеопамяти;
результаты могут немного отличаться из-за порядка вычислений float32.

Состояние оптимизатора и генераторов случайных чисел восстанавливается, но
результаты CPU и GPU могут отличаться из-за порядка вычислений и округления.
GPU-вычисления также не гарантируют побитовую повторяемость между запусками.

## Данные

Оба встроенных корпуса англоязычные. Байтовый токенизатор поддерживает
русский UTF-8, но для русской генерации нужны русскоязычные данные.

| Корпус | Исходный объём | Происхождение и условия источника |
| --- | ---: | --- |
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md) | 1.924 ГБ train + 19.4 МБ validation | Синтетические рассказы Ronen Eldan и Yuanzhi Li; авторы указывают CDLA-Sharing-1.0. |
| [Tiny Shakespeare](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/Readme.md) | 1.12 МБ | Пьесы William Shakespeare, корпус Andrej Karpathy; репозиторий источника использует MIT. |

`download` сохраняет текст в `data/raw/<dataset>/`, а рядом —
`SOURCE_README.md` и `manifest.json` с URL, закреплённой ревизией, размером,
SHA256 и атрибуцией. SDK Hugging Face и API-ключи не нужны.
Основные файлы сверяются с известными upstream SHA256. Загрузка публикует
готовый файл атомарно; `.part` позволяет продолжить прерванное скачивание.

`prepare` создаёт `data/tokenized/<dataset>/train.bin`, `val.bin` и
`metadata.json`. Формат — little-endian uint16: BOS, байты UTF-8, EOS для
каждого документа. TinyStories сохраняет официальные train/validation;
Shakespeare делится по параграфам: первые 90% в train, последние 10% в validation.
Окна могут пересекать EOS/BOS, отдельной attention-маски между документами нет.

Текст TinyStories читается по рассказам, бинарники — через mmap.
Токенизация примерно удваивает объём UTF-8 текста. Один русский символ
обычно занимает несколько токенов; некорректные UTF-8 последовательности
при генерации заменяются символом `�`. Корпуса, `.venv` и checkpoints
исключены из Git.

## Сохранение и продолжение

`checkpoint.npz` — собственный формат ZIP с `.npy` и JSON, без `pickle`.
Он содержит веса, архитектуру, гиперпараметры, шаг, моменты Adam, состояния
RNG train/validation и SHA256 данных. Инференс не загружает моменты Adam.

Продолжение примера 250М с последнего сохранённого состояния:

```powershell
uv run microllm train --resume runs/250m/checkpoint.npz --data-dir data/tokenized/tinystories --output-dir runs/250m
```

При `--resume` конфигурация берётся из checkpoint. Изменение архитектуры,
данных или гиперпараметров, включая общий горизонт `--steps`, отклоняется:
это сохраняет learning-rate schedule и последовательность выборки данных.
Побитовая повторяемость вычислений на GPU не гарантируется.
Для запланированной ранней остановки добавьте к исходной команде
`--stop-after N`: N — абсолютный номер шага, общий `--steps` не меняется.

Checkpoint пишется каждые `--save-every` шагов и при нормальном завершении.
**Ctrl+C не сохраняет текущий шаг**: продолжить можно с последнего готового
checkpoint. Выбирайте разумный интервал, например 100 шагов для маленькой
модели и 1000 для большой, учитывая стоимость записи. Один output-dir
используйте для одного прогона; запуск без `--resume` начинает со случайных
весов и заменяет checkpoint.

В `metrics.jsonl` поле `train_loss` относится к текущему пакету до обновления,
`val_loss` — к отдельным validation-пакетам после обновления. Это выборочные
оценки, а не полный расчёт по корпусу.

CLI выводит события `structlog` в `stderr` с цветом и временем UTC: этапы
загрузки, шаги обучения и сохранение checkpoint. Для JSON-логов добавьте
`--log-format json` перед командой, например `microllm --log-format json train`.
Результаты команд `info`, `generate`, `download` и `prepare` выводятся в
`stdout`. Формат файла `metrics.jsonl` при этом не меняется.

## Проверки

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

Тесты проверяют конечные разности для каждого массива весов, причинность
attention, снижение loss, AdamW, точное resume с accumulation, RNG и моментами,
целостность данных, токенизацию, sampling и формат checkpoint. Функции имеют
тайпхинты; управляющие блоки отделены пустыми строками.

При разработке маленькая модель после 500 шагов дала train loss 5.546 → 2.619
и финальную выборочную validation loss 2.551. Отдельно проверялся один шаг
полной модели 250М: forward, ручной backward, AdamW, сохранение и загрузка
для инференса. Это исторические проверки работоспособности, а не результаты
полного предобучения; их локальные веса и метрики в репозиторий не включены.

## Лицензия

Код распространяется под [GNU GPL v3.0](LICENSE).
Датасеты имеют собственные условия использования, указанные в ссылках
на источники выше; лицензия кода их не заменяет.
