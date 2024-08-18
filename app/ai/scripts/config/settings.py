import json
from typing import Optional, List

from pydantic import ConfigDict, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_weights_path: str = "../models/weights.pt"
    model_metadata_path: str = "../models/metadata.json"
    train_data_path: str = "../data/names/*.txt"
    environment: str = 'develop'  # "develop" | "prod"

    n_letters: Optional[int] = None
    n_hidden: Optional[int] = None
    n_categories: Optional[int] = None
    all_categories: Optional[List[str]] = []

    @model_validator(mode="after")
    def data_conversion(self):
        # fill from models/metadata.json after train

        # for local testing (cli & Docker)
        if self.environment == 'develop':
            with open(self.model_metadata_path, "r") as f:
                rnn_params = json.load(f)
                self.__dict__.update(rnn_params)

        if self.environment == 'prod':
            # For Kubernetes deployment
            self.__dict__.update({
              "n_letters": 58,
              "n_hidden": 128,
              "n_categories": 18,
              "all_categories": [
                "Czech",
                "German",
                "Arabic",
                "Japanese",
                "Chinese",
                "Vietnamese",
                "Russian",
                "French",
                "Irish",
                "English",
                "Spanish",
                "Greek",
                "Italian",
                "Portuguese",
                "Scottish",
                "Dutch",
                "Korean",
                "Polish"
              ]
            })

        return self

    model_config = ConfigDict(
        protected_namespaces=('settings_',)
    )

settings = Settings()