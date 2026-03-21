# MLOps Vehicle Insurance Claims

CLI-интерфейс для загрузки данных, обучения и валидации моделей регрессии по страховым выплатам.

## Структура команд

```text
cli()
├── --clear -> clear current session, 
├── add_data -> adding new chunk of data 
├── train
│   ├── данные
│   ├── --new -> создать модель, обучить на всех данных, сохранить
│   └── --old -> загрузить модель, дообучить, сохранить
└── val
    ├── данные
    ├── --new -> создать модель, train/test split, обучить, оценить
    └── --old -> загрузить модель, оценить на всех данных
```

## Базовый синтаксис

```bash
python run.py --mode <train|val|add_data> [опции]
```
