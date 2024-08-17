from typing import List

from fastapi import APIRouter
from pydantic import Field, BaseModel, model_validator

from ai.scripts.config.settings import settings
from ai.scripts.predict import predict

router = APIRouter()


class ModelPrediction(BaseModel):
    value: float = Field(...)
    category: str = Field(...)

    @model_validator(mode="before")
    @classmethod
    def data_conversion(cls, values):

        values["value"] = float(values["value"])

        return values


class ModelResponse(BaseModel):
    res: List[ModelPrediction]


@router.post("/")
async def analyze(
        name: str,
        n: int = 3
) -> ModelResponse:
    # "-n", type=int, default=3, help="return top n mosty likely classes"
    # "name", type=str, help="name to classify"

    output = predict(name, n, settings.model_weights_path)

    return ModelResponse(res=output)
