from database import Base
from models import User
from sqlalchemy.orm import Session
from schemas import UserRegister

def get_users_and(db: Session, username: str="%", email: str="%", first: bool=False):
    """
    Queries a database using a Session object given some criteria with an AND filter;
    if multiple query values are passed, rows are only returned if all conditions are met.
    If a query parameter is left blank it is not part of the filter.

    Inputs:
        -db: Session, the Session object to query the database through
        -username: str, the username to query by
        -email: str, the email address to query by
        -first: bool, if True, will return only the first result of the query. Returns all otherwise.
    Outputs:

    """

    if first:
        return db.query(User).filter(
            User.username.like(username),
            User.email.like(email)).first()
    else:
        return db.query(User).filter(
            User.username.like(username),
            User.email.like(email)).all()

def create_user(db: Session, user_registration: UserRegister):

    user = User(**user_registration.dict())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def delete():

    pass