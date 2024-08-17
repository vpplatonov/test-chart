import uvicorn
import fastapi
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware

from ai.scripts.config.settings import settings
from ai.scripts.predict import model_load

from api.routes.predict import router

# disable OpenAPI (swagger) for production
if settings.environment == 'prod':
    app = fastapi.FastAPI(openapi_url=None)
else:
    app = fastapi.FastAPI()

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

@app.on_event("startup")
async def startup_event():
    # Made Model hot
    model_load(weights_file=settings.model_weights_path)


app.include_router(router, prefix='/api/predict', tags=['predict'])
app.add_middleware(SessionMiddleware, secret_key="!secret")


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)
