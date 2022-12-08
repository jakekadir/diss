from schemas import UserInDB, UserRegister, Token, UserDBQuery, RelationshipType
import schemas
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta, datetime
from pydantic import EmailStr, ValidationError
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from models import User, UserRelationship
from dependencies import get_password_hash, authenticate_user, create_access_token, get_current_active_user, get_db
import crud

router = APIRouter()

@router.post("/login", response_model=Token)
async def login(db: Session = Depends(get_db), form_data: OAuth2PasswordRequestForm = Depends()):
    # authenticate user by checking if user exists and compare password hashes
    user = authenticate_user(db, username=form_data.username, password=form_data.password)
    
    # raise appropriate errors
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # create timedelta
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    # creates access token using username to expire in ACCESS_TOKEN_EXPIRE_MINUTES mins
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register(password: str = Form(), email = Form(), username = Form(), db: Session=Depends(get_db)):
    
    # create hash from password
    hash = get_password_hash(password)

    # create user model for database
    user: UserRegister = UserRegister(email=email, 
                                        hashed_pass=hash,
                                        username=username,
                                        disabled=False)

    # add user to database
    created_user = crud.create_user(db, user)

    return created_user