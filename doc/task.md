# Техническое описание задачи и реализации (MLOps Vehicle Insurance Claims)

Все пути и ключи конфига приведены относительно корня репозитория; актуальная схема конфигурации — файл `config.yaml`.

---

## 1. Постановка задачи и домен

### 1.1. Бизнес-смысл

Репозиторий эмулирует **MLOps-пайплайн** для страховых **vehicle insurance claims**: в систему поступают **батчи** наблюдений (в учебной постановке — годовые CSV по дате начала полиса `INSR_BEGIN`), данные нормализуются в **SQLite**, по ним строятся отчёты качества и при необходимости дрейфа, затем обучаются регрессионные модели, предсказывающие **размер выплаты по урегулированному убытку**.

### 1.2. Что прогнозируется

- **Целевая переменная** (регрессия): колонка **`CLAIM_PAID`** — сумма выплаты по претензии.
- В конфиге: `columns.target: "CLAIM_PAID"`.
- Распределение цели **сильно zero-inflated** (много нулевых выплат при наличии полиса), из-за чего метрика **R²** на сырых масштабах может оставаться низкой; в коде моделей цель **логарифмируется** при обучении (см. раздел 6).

### 1.3. Идентификатор и время

- **`OBJECT_ID`** — идентификатор объекта (`columns.id`).
- **`INSR_BEGIN`** — дата события для фильтрации выборок при обучении/валидации из БД (`columns.datetime`, формат `columns.datetime_format`: `%d-%b-%y`).
- Колонка **`INSR_END`** исключается из признаков (`columns.drop`).

---

## 2. Источник данных и эмуляция потока

### 2.1. Датасет

Используются открытые данные **Ethiopian Motor Insurance** (2011–2018). В репозитории они разбиты на файлы `datasets/motor_data_<год>.csv` (см. таблицу в корневом `README.md`).

### 2.2. Конфигурация источников

Блок **`data_sources`** в `config.yaml` задаёт список CSV для **первичной** загрузки в БД (при пустой базе или при режиме `--drift-ref` / `ensure_db`).

### 2.3. Поток в систему

1. **`run.py --mode add_data --path-csv <файл>`** — добавление **одного** нового батча: функция `db_add_tables` (`src/data/database/db_create.py`) стримит CSV батчами размера **`batch.size`** (32768), сортируя по **`batch.sort_by`** (`INSR_BEGIN`), пишет строки в таблицу **`raw_events`** (поле `raw_json` — JSON строки исходного набора полей).
2. **`ensure_db`** (перед `train` / `val` / `analyse`) — если БД отсутствует или пуста, загружаются все файлы из `data_sources` с полным quality-пайплайном (`run_quality=True` по умолчанию в этом пути).

Путь к SQLite: **`data_storage.data_path`** → `session/data/db_sqlite.db`.

---

## 3. Признаковое пространство (конфиг)

### 3.1. Числовые признаки

Из `config.yaml`, ключ `columns.features.numeric`:

`EFFECTIVE_YR`, `INSURED_VALUE`, `PROD_YEAR`, `SEATS_NUM`, `CARRYING_CAPACITY`, `CCM_TON`, `PREMIUM`.

### 3.2. Категориальные признаки

`SEX`, `INSR_TYPE`, `TYPE_VEHICLE`, `MAKE`, `USAGE`.

### 3.3. Использование в коде

Список признаков для диагностик (например, flexible auto) собирается утилитой **`get_all_features`** (`src/data/utils/utils.py`): числовые + категориальные из конфига.

---

## 4. Data Quality (качество данных)

### 4.1. Роль артефакта `db_quality.yaml`

Файл **`session/reports/db_quality.yaml`** (`data_storage.quality_path`) — центральный **контракт допустимых имён и типов признаков** для последующего препроцессинга: при сборке матрицы для модели проверяется согласованность схемы с этим отчётом (см. `src/preprocessing/`).

### 4.2. Глобальная статистика и мета

- **`session/reports/db_statistics.yaml`** — агрегированная статистика по числовым/категориальным колонкам и мета-метрики.
- **`session/reports/db_meta.yaml`** — метаданные загрузок.

### 4.3. Пороги качества

В `quality.stats_thresholds` задаются пороги для missing/nonvalid/zero frequency, CV числовых колонок, доминирования категорий и т.д. (см. `config.yaml`). Округление в отчётах: **`quality.round_precision`**.

### 4.4. Пайплайн обновления

После **`add_data`** вызывается **`refresh_quality_artifacts`** (`src/data/quality/pipeline.py`) при `run_quality=True`. Отдельно в Docker/README документирован запуск **`python -m src.data.quality.pipeline`** для полной пересборки артефактов в чистой `session/`.

---

## 5. Ассоциативные правила и feature engineering на уровне строк

Блок **`quality.association`** управляет **Apriori** (через **mlxtend**) на транзакциях из бинаризованных/дискретизованных признаков:

- **`max_transactions`**, **`sample_cap_per_column`**, **`max_levels_per_categorical`** — ограничения по памяти и размерности.
- **`n_bins`**, **`binning`** (quantile) — дискретизация числовых колонок для mining.
- **`min_support`**, **`min_confidence`**, **`min_lift`**, **`top_k`** — параметры правил.
- **`add_rule_features`**, **`max_rule_features`** — добавление до N бинарных признаков-правил в сырые строки перед ML-препроцессингом (см. цепочку `apply_feature_engineering_rows` в загрузке train).

Исключения: **`exclude_target`**, список **`exclude_from_association`**.

---

## 6. Дрейф данных (Data Drift)

### 6.1. Эталон и отчёт

- **`quality.drift.reference_path`** — `session/reports/drift_reference.yaml` (эталонная статистика).
- **`quality.drift.report_path`** — `session/reports/drift_report.yaml` (текущее сравнение с эталоном).

### 6.2. Когда пересчитывается отчёт

Функция **`run_drift_monitor`** (`src/data/quality/drift.py`) вызывается из **`db_add_tables`** только если **`quality.drift.run_check_after_add_data: true`**. Сравниваются текущие агрегаты с эталоном; пороги — `mean_shift_sigma_*`, `std_ratio_*`, `missing_frequency_delta_*`, `categorical_jsd_*`; политика остановки — **`fail_on`**, **`fail_on_incomplete`**.

### 6.3. Режим `--drift-ref`

`run.py --drift-ref` (без `--mode`): **`build_drift_reference`** пересоздаёт БД из всех `data_sources` с отключённым полным quality refresh внутри этого шага, затем **`freeze_drift_reference`** копирует статистику в эталон. После этого CLI снова вызывает **`db_clear`** — **очищается вся `session/`**; команда для «снять эталон и начать с чистого листа».

**Важно:** обычный **`train`** сам по себе **`drift_report.yaml` не обновляет** — только цепочка `add_data` / отдельный quality pipeline.

---

## 7. EDA

- Режим **`run.py --mode analyse`**: выборка из БД, отчёт **ydata-profiling**.
- Параметры: **`quality.eda`** — `report_path` (`session/reports/eda_profile.html` по умолчанию), `max_rows`, `minimal_profile`, `title`.

---

## 8. Препроцессинг для моделей (`src/preprocessing`)

### 8.1. Два уровня преобразований

1. **На сырых строках** (до матрицы): `preprocessing.feature_engineering` — **`log1p`** для указанных колонок, **отношения** (`ratios`), **разности** (`differences`), плюс ассоциативные признаки из quality (если включены).
2. **Матрица для модели**: классы, наследующие **`BasePreprocessor`**, варианты из **`preprocessing.variants`** — для каждого варианта задаются стратегии **impute** (median / most_frequent), **scale** (да/нет), **encode** (**ordinal** vs **onehot**) отдельно для числовых и категориальных колонок.

### 8.2. Варианты в конфиге

| Ключ варианта | Назначение |
|---------------|------------|
| **`catboost_ord`** | median impute, без scaling, ordinal категории — типичный ввод для CatBoost с категориальными признаками в DataFrame |
| **`mlp_ohe`** | median, scale, one-hot — разреженная числовая матрица для MLP |
| **`mlp_ord`** | median, scale, ordinal — компактнее, чем OHE |

**`preprocessing.default_variant`** — вариант по умолчанию, если не включён перебор.

### 8.3. Перебор вариантов

**`preprocessing.tune_preprocess_variants: true`** — при `train` для каждого кандидата варианта строится матрица, обучается модель выбранного семейства, выбирается вариант с **лучшим validation RMSE**, затем финальный `fit` на полной выборке с победившим вариантом (см. `run.py`, `train_call`).

### 8.4. Пропуски в цели

**`preprocessing.target_missing_fill`** — подстановка для пропусков цели (в конфиге `0`).

---

## 9. Проектирование моделей

### 9.1. Две модели + автовыбор семейства

| Режим CLI | Реализация | Файл |
|-----------|------------|------|
| `--new catboost` | **CatBoost** `CatBoostRegressor` | `src/training/models/catboost_regressor.py` |
| `--new mlp` | **sklearn** `MLPRegressor` | `src/training/models/nn_regressor.py` |
| `--new auto` | Эвристика по **сырым** train-строкам → выбор `catboost` или `mlp` | `src/training/models/flexible_model.py` |

Эвристика **не читает EDA-HTML**: считает долю строк с любым пропуском по полному списку признаков, долю «выбросов» по IQR на числовых колонках, порог по числу строк; пороги переопределяются опциональным блоком **`training.flexible_model`** в YAML (ключ **`enabled`** в merge игнорируется — включается только CLI `auto`).

### 9.2. Общая оболочка — `BaseRegressor`

Файл **`src/training/models/base.py`**:

- **Цель**: \(y_{\text{train}} = \log(1 + y)\) (`_transform_target`).
- **Смещение при обратном преобразовании**: `expm1(pred + resid_var/2)` — эвристика с дисперсией остатков в лог-пространстве (`_resid_var` после fit на train).
- **Метрики** `evaluate`: **RMSE**, **RMSLE** (на `log1p` с клипом предсказаний), **R²** на исходной шкале выплат.
- **Валидация при `train`** (аргумент `validation` из `training.validation` в конфиге):
  - **`holdout`**: `train_test_split`, доля теста `test_size`, `random_state`, затем fit на train-части лога цели и метрики на holdout.
  - **`kfold`**: `KFold` (`n_splits`, `shuffle`, `random_state`); на каждом фолде — **свежий клон** модели (`_fresh_clone`), метрики усредняются; затем **полный refit** на всех данных.
  - **`time_series` / `timeseries`**: `TimeSeriesSplit`; при слишком малой выборке — откат на holdout.
- **`update`**: донастройка с `continue_training=True` (CatBoost через `init_model`, MLP через `warm_start`), затем `evaluate` на переданной выборке.

### 9.3. Гиперпараметры CatBoost

Базовые дефолты в коде (`CatBoostRegressionModel.__init__`): `loss_function=RMSE`, `verbose=False`, `random_seed=42`, `iterations=500`; далее **`kwargs.update`** из блока **`models.catboost`** в `config.yaml`:

- В текущем конфиге: `loss_function`, `verbose`, `random_seed`, `iterations` (дублируют/перекрывают дефолты при совпадении ключей).

При `fit` передаются **`cat_features`** из колонок DataFrame (`run.py` → `cat_features_from_frame`).

### 9.4. Гиперпараметры MLP (sklearn)

Параметры конструктора **`MLPRegressionModel`** (и сохраняемые в `_init_json` для клонирования):

- **`hidden_layer_sizes`** — по умолчанию `(64, 32)`, из YAML список `[64, 32]` приводится к tuple.
- **`lr`** → `learning_rate_init`, **`max_epochs`** → `max_iter`, **`batch_size`**, **`patience`** → `n_iter_no_change`, **`val_fraction`** → `validation_fraction`, **`random_state`**, **`alpha`** (L2).
- **`loss`**, **`huber_delta`**: сохраняются в **`_init_kwargs`** для клонирования/сериализации обёртки; в **`_build_model`** в `MLPRegressor` они **не передаются** — используются стандартные параметры sklearn (`activation="relu"`, `solver="adam"`, **`early_stopping=True`** и перечисленные выше). При необходимости Huber-loss потребовалась бы доработка (например, `SGDRegressor` или кастомный шаг).
- Входы **`fit`/`predict`**: приведение к **`float32`** для стабильности.

Блок **`models.mlp`** в YAML задаёт перечисленные гиперпараметры.

### 9.5. Где лежит код

Все классы регрессоров и `flexible_model` — в **`src/training/models/`**; мониторинг метрик модели и профайлер — **`src/training/monitoring/`**.

---

## 10. Обучение, дообучение, валидация (`run.py`)

### 10.1. Источник матрицы при `train` / `val`

Ровно один из:

- **`--path-csv`** — загрузка из файла, feature engineering + association rules, затем препроцессор.
- **`--date-until <ISO-date>`** — строки из SQLite с `event_date <= date` (очищенные батчи + тот же FE).

### 10.2. Сохранение бандла

После `train` в **`model_storage.models_path`** (`session/models/`) пишется **`pickle`** со словарём:

- **`model`** — обученный регрессор;
- **`preprocessor`** — fitted preprocessor (вариант, `model_kind`);
- **`variant`**, **`model_name`**, **`metrics`**;
- **`tune_preprocess_variants`** — флаг, что выбор варианта был свипом;
- **`flexible_selection`** — если был `--new auto` (запрошенное `auto`, выбранное семейство, причина, диагностика).

Имя файла: **`<model_name>_<variant>_<YYYYMMDD_HHMMSS>.pkl`**.

### 10.3. Incremental (`update`)

- CLI: **`--mode train`** без **`--new`**, но с **`--old <bundle>`** или с **`incremental_training.enabled: true`** и **`parent_model`** в конфиге.
- Вызывается **`model.update(X, y)`** на матрице из новых данных с тем же препроцессором; новый бандл с полем **`parent_model`**.

### 10.4. `val`

Загрузка бандла, **`build_val_dataset`** с тем же препроцессором, **`model.evaluate`**, затем **`record_val_model_drift`** — YAML-отчёт и история (см. ниже).

---

## 11. Мониторинг дрейфа метрик модели (Model Drift)

Настройки: **`training.model_drift`** в `config.yaml`.

- **`enabled`** — включает запись отчёта и истории.
- **`metric`** — например `RMSE` (сравнение текущего `val` с метриками из бандла при train).
- **`warn_ratio`**, **`critical_ratio`** — пороги для **stress ratio** (для RMSE/RMSLE: отношение текущего к базовому, ≥1 хуже; для R² — отдельная ветка в `_stress_ratio`).
- **`fail_on`**: `warn` | `critical` | `null` — при нарушении кидается **`ModelDriftPolicyError`** (обрабатывается в `run.py`).
- **`report_path`**, **`history_path`** — куда писать YAML.

**Замечание по конфигу:** в репозитории может быть задано **`critical_ratio: 0.1`** (меньше единицы) — для RMSE это означает **очень жёсткий** порог «текущий не более 10% от базового»; имеет смысл сверить с намерением (часто для «хуже в 1.25 раза» используют значения > 1).

После **`train`** при включённом model_drift вызывается **`append_metrics_history_entry`** — дописывается строка в историю метрик.

---

## 12. Профилирование (опционально)

Блок **`training.profiler`**:

- **`enabled`**, **`time`** (pyinstrument → `time.html`), **`memory`** (memray → `memray.bin` + `memory_flamegraph.html`), **`output_dir`**, **`memray_native_traces`**, **`time_sample_interval`**.
- Каждый прогон — подпапка **`session/reports/profiles/<mode>_<timestamp>_µs/`** и дописка в **`manifest.yaml`**.
- Вложение: **memray снаружи**, **pyinstrument внутри** (иначе конфликт `threading.setprofile`).

---

## 13. Логирование

- **`logging.path`**, **`logging.level`** в конфиге → файл **`session/logs/run.log`** (обработчик настраивается в `run.py`).

---

## 14. Сводка артефактов `session/`

| Путь | Содержание |
|------|------------|
| `session/data/db_sqlite.db` | Сырые события |
| `session/reports/db_*.yaml` | Статистика, мета, качество |
| `session/reports/drift_*.yaml` | Эталон и дрейф **данных** |
| `session/reports/eda_profile.html` | EDA |
| `session/reports/model_drift_report.yaml`, `model_metrics_history.yaml` | Дрейф **метрик модели** |
| `session/reports/profiles/` | Профайлер |
| `session/models/*.pkl` | Бандлы |
| `session/logs/run.log` | Логи CLI |

---

## 15. Связь с оценкой по курсу

Распределение баллов и ответственность по блокам курса — в файле **`doc/grade.md`**. Настоящий документ **`doc/task.md`** детализирует реализацию под эти блоки (данные, качество, препроцессинг, две модели, валидация, дрейф, инкремент, CLI, Docker — см. корневой `README.md`).
