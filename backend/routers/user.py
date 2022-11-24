from models import User, UserInDB, UserRegister, Token
from database import user_db
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta, datetime
from pydantic import EmailStr, ValidationError
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from dependencies import get_password_hash, authenticate_user, create_access_token, get_current_active_user

router = APIRouter()

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):

    # build object from the user details from the request to parse the email
    try:
        email: EmailStr = EmailStr(form_data.username)
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email address.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # authenticate user by checking if user exists and compare password hashes
    user = authenticate_user(email, form_data.password)
    
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
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register(password: str = Form(), email = Form()):
    
    # create hash from password
    hash = get_password_hash(password)

    # create user model for database
    user: UserRegister = UserRegister(_email=email, 
                                        hashed_password=hash, 
                                        friends=[],
                                        savedRecipes=[],
                                        starredRecipes=[],
                                        uploadedRecipes=[],
                                        date=datetime.utcnow(),
                                        disabled=False)

    # add user to database
    user_db.add(user)



# protected route
@router.get("/users/me/", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user


@router.get("/users/me/friends/")
async def read_own_friends(current_user: UserInDB = Depends(get_current_active_user)):
    return [{"friends": current_user.friends, "owner": current_user.username}]

