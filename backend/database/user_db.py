from database.db import db
from models import UserInDB
from pydantic import EmailStr

def add(user: UserInDB):

    # check if email already exists
    if get(user.email):
        
        pass

    created_user = db["users"].insert_one(user.dict())

    return created_user

def delete(email: EmailStr):

    pass

def update():

    pass

def get(email: EmailStr) -> UserInDB:

    user = UserInDB(**db["users"].find_one({
        "email" : email
    }))

    return user
