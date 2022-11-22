from models import User, DBUser, Token
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from dependencies import get_password_hash, authenticate_user, create_access_token, get_current_active_user

router = APIRouter()

@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):

    # build object from the user details from the request to parse the email
    request_user = User(email=form_data.username)

    # authenticate user by checking if user exists and compare password hashes
    user = authenticate_user(request_user.username, form_data.password)
    
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
async def register(username: str = Form(), password: str = Form(), email = Form()):
    hash = get_password_hash(password)

    user: DBUser = DBUser(username=username, hashed_password=hash, email=email)

    fake_users_db[username] = user.dict()

    print(fake_users_db)

# protected route
@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@router.get("/users/me/items/")
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return [{"item_id": "Foo", "owner": current_user.username}]

