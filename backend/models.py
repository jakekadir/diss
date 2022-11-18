import uuid
from typing import Optional, List
from pydantic import BaseModel, Field, EmailStr, HttpUrl, UUID4

class User(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4, alias="_id")
    email: UUID4 = Field(alias="_email")
    friends: List[UUID4]


    # defines example record
    class Config:
        schema_extra = {
            "example" : {
                "id" : 1,
            }
        }

class Recipe(BaseModel):
    id: UUID4 = Field(default_factory=uuid.uuid4, alias="_id")
    title: str
    steps: List[str]
    author_id: str
    author_name: str
    cook_time: str
    prep_time: str
    total_time: str
    date_published: str
    description: str
    image_urls: List[HttpUrl]
    saturated_fat_content: float
    cholesterol_content: float
    sodium_content: float
    carbohydrate_content: float
    fiber_content: float
    sugar_content: float
    protein_content: float
    recipe_servings: float
    recipe_yield: str 