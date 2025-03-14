from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к обученной модели
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model")

logger.info("Loading model")
# Загружаем токенизатор и модель
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Создаем pipeline для классификации
pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device="cpu"  # Используй "cuda" если есть GPU
)

logger.info("Model loaded")