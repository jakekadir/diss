from typing import List, Union, Optional
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from enum import Enum
from models import RelationshipType

class Relationship(BaseModel):
    sender_id: int
    recipient_id: int
    status: RelationshipType
    
class User(BaseModel):
    email: EmailStr = Field()
    username: str
    # relationships: List[Relationship]
    # savedRecipes: List[int]
    # starredRecipes: List[int]
    # uploadedRecipes: List[int]
    disabled: bool

    class Config:
        orm_mode = True

class UserRegister(User):
    hashed_pass: str

class UserInDB(UserRegister):
    id: int

class UserDBQuery(BaseModel):
    email: Optional[EmailStr]
    id: Optional[int]
    username: Optional[str]

    class Config:
        orm_mode = True


class Recipe(BaseModel):
    id: int
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

# token model
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Union[EmailStr, None] = None