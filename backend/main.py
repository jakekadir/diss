from fastapi import FastAPI
from routers.user import router as user_router
from routers.recipe import router as recipe_router
from routers.relationships import router as relationship_router
from database import engine
import models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_router)
app.include_router(relationship_router)
app.include_router(recipe_router)
