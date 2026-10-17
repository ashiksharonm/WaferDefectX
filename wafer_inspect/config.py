from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_PATH: str = "models/wafer_model.joblib"
    METRICS_PATH: str = "models/metrics.json"
    QUEUE_DB_PATH: str = "data/review_queue.sqlite3"
    SAMPLES_DIR: str = "samples"
    # Wafers whose calibrated top-class probability is below this go to the review queue.
    REVIEW_CONFIDENCE_THRESHOLD: float = 0.8
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    LOG_LEVEL: str = "INFO"


settings = Settings()
