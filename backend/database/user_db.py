from database.db import db
from models import UserInDB, UserDBQuery
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

def get(user_query: UserDBQuery) -> UserInDB:

    query_result = db["users"].find_one(
        user_query.dict()
    )

    if query_result is not None:
        user = UserInDB(**query_result)

        return user
