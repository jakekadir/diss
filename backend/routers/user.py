from schemas import UserInDB, UserRegister, Token, UserDBQuery, Relationship, RelationshipType
from config import ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta, datetime
from pydantic import EmailStr, ValidationError
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from models import User
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
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register")
async def register(password: str = Form(), email = Form(), username = Form(), db: Session =Depends(get_db)):
    
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
    

"""
options:
    - each user stores several arrays:
        - friends
        - sent request
        - received request
        - denied request
        - blocked (only appears in the record of the user who has blocked the other)
    - each user stores an object with:
        - friend ID
        - relationship type
            - friendship
            - received request
            - sent request
            - denied request
            - blocked
    - separate table with fields:
        - initiating user
        - subject user
        - relationship type
            - friendship
            - sent request
            - denied request
            - blocked request
"""


"""
friendship routes
"""
@router.post("/users/send-friend-request")
async def send_friend_request(current_user: UserInDB = Depends(get_current_active_user), 
                                friend_username: str = Form()):
        
    # query database for username
    user_query: UserDBQuery = UserDBQuery(username=friend_username)
    
    # friendDB: UserInDB = user_db.get(user_query)


    # # check user exists
    # if friendDB is None:
    #     raise HTTPException(
    #                 status_code=status.HTTP_404_NOT_FOUND,
    #                 detail="Cannot find user to send friend request to."
    #             )

    # # if current user has no friend objects for the requested friend:
    # if friend_username not in [relationship.id for relationship in current_user.relationships]:

    #     # create friend object for current user
    #     relationship: Relationship = Relationship(id=friendDB.id, status=RelationshipType.SENT)
             
        # update record

    # if user already has a relationship with the requested friend:
    """
        match relationship type:

            case RECEIVED:

                update as friends

                update friend's friend object

            case BLOCKED:

                change blocked attribute to sent

            case 
    
    """
    # if not blocked
    
        # create friend object for friend

        # update record

# test route
@router.get("/tmp")
async def tmp(db: Session = Depends(get_db)):

    user = UserRegister(
        email="jakekadir0@gmail.com",
        username="jjsmithson",
        disabled=False,
        hashed_pass="kjasdkjaskj")

    crud.create(db, user)

@router.get("/tmp2")
async def tmp2(db: Session = Depends(get_db)):

    crud.get_users(db)

@router.get("/tmp3")
async def tmp3(db: Session = Depends(get_db), current_user: UserInDB = Depends(get_current_active_user)):

    friend = crud.get_users_and(db, username="jjsmithson")

    current_user.add_friend(friend)


# protected route
@router.get("/users/me/", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user


@router.get("/users/me/friends/")
async def read_own_friends(current_user: UserInDB = Depends(get_current_active_user)):
    return [{"friends": current_user.relationships, "owner": current_user.username}]

