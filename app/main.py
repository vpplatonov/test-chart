from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from ai.scripts.config.settings import settings
from ai.scripts.predict import model_load
from api.routes.predict import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Load the ML model. Replaced deprecated @app.on_event("startup") """
    model_load(weights_file=settings.model_weights_path)
    yield
    # Clean up the ML models and release the resources
    model_load.cache_clear()


# disable OpenAPI (swagger) for production
if settings.environment == 'prod':
    app = FastAPI(
        openapi_url=None,
        lifespan=lifespan
    )
else:
    app = FastAPI(lifespan=lifespan)

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix='/api/predict', tags=['predict'])
app.add_middleware(SessionMiddleware, secret_key="!secret")


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
