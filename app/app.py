# Логирование
import logging

# FastAPI и обработка ошибок
from fastapi import FastAPI, HTTPException

# Валидация входных и выходных данных
from pydantic import BaseModel

# Загрузка предобученной модели
from app.load_model import pipe

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Описание API
description = """
API для анализа сарказма в текстах.

## Как использовать:
- Отправьте текст в POST-запросе на `/sentiment`
- В ответе будет предсказанная метка (`SARCASM` или `NOT_SARCASM`) и вероятность.

"""

# Создание объекта FastAPI
logger.info("Запуск FastAPI")
app = FastAPI(
    title="Sarcasm Analysis API",
    description=description,
    summary="Определение сарказма в текстах",
    version="1.0.0"
)

# Входные данные
class UserInput(BaseModel):
    user_input: str

# Выходные данные
class ModelOutput(BaseModel):
    label: str
    score: float

# Корневой маршрут
@app.get("/")
async def root():
    return {"message": "Добро пожаловать в Sarcasm Analysis API!", "docs": "/docs"}

# Основной эндпоинт предсказаний
@app.post("/sentiment", response_model=ModelOutput)
async def predict_answer(user_input: UserInput):
    try:
        text = user_input.user_input
        output = pipe(text)
        return ModelOutput(label=output[0]['label'], score=output[0]['score'])
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        raise HTTPException(status_code=500, detail=str(e))
