from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_weights_path: str = "../models/weights.pt"
    model_metadata_path: str = "../models/metadata.json"
    train_data_path: str = "../data/names/*.txt"
    environment: str = 'dev'  # "dev" | "prod"

    model_config = ConfigDict(
        protected_namespaces=('settings_',)
    )

settings = Settings()