from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import User

router = APIRouter()

@router.post("/", response_description="Create a new user",
                status_code=status.HTTP_201_CREATED, response_model=User)
def create_user(request: Request, user: User = Body()):
    user = jsonable_encoder(user)