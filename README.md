# Sarcasm - analysis

Этот репозиторий содержит обученную модель `Hugging Face BERT` на корпусе англоязычных текстов.

Из-за лимитов GitHub папка model/ доступна по ссылке: https://disk.yandex.ru/d/XBZcTr1kJ04Maw

## Code

Предобученная модель находится здесь: [Hugging Face BERT](https://huggingface.co/docs/transformers/en/model_doc/bert).

Для обучения используется обычный Trainer.

Для развертывания модели используется Pipeline и FastAPI, стандартный порт `1234`.

## Expertiments

Исползуемые ресурсы:
- WSL (Ubuntu 22.04)
- RTX 4090
- 16 RAM в WSL
- Ryzen 5900x (12 ядер в WSL)

Данные:
- дисбаланса нет


Что использовалось:
- TF-IDF + CatBoost/LogisticRegression
- sentence-transformer + Catboost
- TinyBert
- Hugging Face BERT


Предобработка и валидация:
- По стандарту делим датасет на отложенную, обучающую и валидационную в соотношении `0.8/0.2/0.2`.
- В классик ML не будем использовать валидационную выборку.
- В языковых моделях используем всю выборку и проверяем уже на тестовой при валидации в трейнере.
- В качестве метрики - `F1`, так как классы сбалансированы.

Результат:
- Лучший результат показывает модель "Hugging Face BERT", обучается долго, но большое преимущество в том, что показывает хороший результат на тестовых данных


## Deploy and inference

Собранный образ основан на базовом Python. Новый образ нужно собрать с помощью `build`.

Для инференса используем Docker-compose:

```bash
docker build -t sarcasm-analysis .
```

После чего можно споконо заходить на Swagger через порт `1234`.

    ## TODO


- [ ] Оптимизация модели
- [ ] Ускорение процесса обучения
- [ ] Автоматический подбор гиперпараметров
- [ ] Docker для обучения-экспериментов и инференса итоговой модели
- [ ] Переписать код на нативный PyTorch