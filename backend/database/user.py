from database.db import db
from pydantic import EmailStr

def add():

    pass

def delete():

    pass

def update():

    pass

def get(email: EmailStr):

    user = db["users"].find_one({
        "email" : email
    })

    print(user)
