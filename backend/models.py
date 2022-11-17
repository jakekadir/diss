import uuid
from typing import Optional
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    email: EmailStr = Field(alias="_email")

# class Recipe(BaseModel):
    