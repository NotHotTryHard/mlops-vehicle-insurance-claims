# MLOps Vehicle Insurance Claims

Предполагается, что данная модель представляет из себя идейную надстройку над некоторым пайплайном получения потоковых данных. То бишь, данные будут поступать в систему батчами длиной, к примеру, в пол года, и сладываться в сыром виде в папку datasets/, то есть функционал загрузки данных в рабочее пространство этой модели предполагается делигированным другому устройству/пайплайну.

В качестве эмуляции такого потока исходные данные (Ethiopian Motor Insurance, 2011–2018) разбиты на годовые батчи по колонке `INSR_BEGIN`. Это СЛОЖНЫЙ датасет с реальными, причем zero-inflated данными, поэтому добиться адекватного показателя R2 (> 0.05 мне не удалось). 

| Файл | Строк |
|---|---|
| `datasets/motor_data_2011.csv` | 69 068 |
| `datasets/motor_data_2012.csv` | 91 803 |
| `datasets/motor_data_2013.csv` | 90 943 |
| `datasets/motor_data_2014.csv` | 107 593 |
| `datasets/motor_data_2015.csv` | 116 015 |
| `datasets/motor_data_2016.csv` | 130 412 |
| `datasets/motor_data_2017.csv` | 138 902 |
| `datasets/motor_data_2018.csv` | 57 300 |

Таким образом, каждый файл — это один "прилетевший батч", который подаётся в систему через `--mode add_data` или через изначальный конфиг.

## Структура команд

Предложенный интерфейс работы с программой отвергаю, как, по моему мнению, не продуманный и не совместимый с функционалом других пунктов задачи, взамен предлагаю следующий CLI-интерфейс для загрузки данных, обучения и валидации моделей.

```text
cli()
├── --clear -> clear current session, 
├── add_data -> adding new chunk of data 
├── train
│   ├── данные: из CSV (--path-csv) или из БД (--date-until)
│   ├── --new -> создать модель, обучить на всех данных, сохранить
│   └── --old -> загрузить модель, дообучить, сохранить
└── val
    ├── данные: из CSV (--path-csv) или из БД (--date-until)
    ├── --new -> создать модель, train/test split, обучить, оценить
    └── --old -> загрузить модель, оценить на всех данных
```

## Базовый синтаксис

```bash
python run.py --mode <train|val|add_data> [опции]
```

## Доступные модели

| Ключ        | Описание                              |
|-------------|---------------------------------------|
| `catboost`  | CatBoost регрессор                    |
| `mlp`       | MLP (PyTorch)                         |

## Хранилище моделей

Каждая обученная модель сохраняется как один файл с timestamp:

```text
session/model_stash/
└── <model_name>/
    └── model.pkl          # сериализованные веса модели
        ├── metrics        # метрики качества
        └── meta           # метаданные версии
```

Имя файла (без `.pkl`) передаётся в `--old` при валидации.

## Примеры использования

```bash
# Загрузить новый батч данных в БД
python run.py --mode add_data --path-csv datasets/motor_data14-2018.csv

# Обучить новую CatBoost модель
python run.py --mode train --path-csv datasets/motor_data14-2018.csv --new catboost

# Оценить сохранённую модель
python run.py --mode val --path-csv datasets/motor_data14-2018.csv --old catboost_20260321_171846

# Очистить БД
python run.py --clear
```

## Установка

Последнее время балдею с [uv](https://github.com/astral-sh/uv). Создадим виртуальную среду с Python 3.10 и установим зависимости:

```bash
uv venv .venv --python 3.10
source .venv/bin/activate      # или .venv\Scripts\activate
uv pip install -r requirements.txt
