import uuid
from typing import List, Union
from pydantic import BaseModel, Field, EmailStr, HttpUrl, UUID4
from bson.objectid import ObjectId

# creates a new Pydantic type wrapper for ObjectId, allowing for use in Pydantic models
class PydanticObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, ObjectId):
            raise TypeError('ObjectId required')
        return str(v)
    
class User(BaseModel):
    email: EmailStr = Field()
    friends: List[UUID4]
    disabled: bool

class UserRegister(User):
    hashed_password: str
    date: str

class UserInDB(UserRegister):
    id: PydanticObjectId = Field(alias="_id")

class Recipe(BaseModel):
    id: PydanticObjectId = Field(alias="_id")
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