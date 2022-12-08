from database import Base
from models import User, UserRelationship
from sqlalchemy.orm import Session
from schemas import UserRegister
from typing import Union, List

def get_users(db: Session, username: str="%", email: str="%", first: bool=True, and_query: bool=True) -> Union[User, List[User]]:
    """
    Queries a database using a Session object given some criteria with an AND filter;
    if multiple query values are passed, rows are only returned if all conditions are met.
    If a query parameter is left blank it is not part of the filter.

    Inputs:
        -db: Session, the Session object to query the database through
        -username: str, the username to query by
        -email: str, the email address to query by
        -first: bool, if True, will return only the first result of the query. Returns all otherwise.
        -and_query: bool, if True will query records with all parameters true. If False, an OR query is performed.
    Outputs:
    """

    if and_query:
        query = db.query(User).filter(
            (User.username.like(username)) &(User.email.like(email))
        )
    else:
        query = db.query(User).filter(
            (User.username.like(username)) | (User.email.like(email))
        )
    return query.first() if first else query.all()

def create_user(db: Session, user_registration: UserRegister):

    user = User(**user_registration.dict())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def delete():

    pass

def get_relationships(db: Session, user_id: str="%", friend_id: str="%", first: bool=True) -> Union[UserRelationship,List[UserRelationship]]:
    """
    Queries a database using a Session object given a user_id and a friend_id.
    Inputs:
        -db: Session, the Session object to query the database through
        -username: str, the username to query by
        -email: str, the email address to query by
        -first: bool, if True, will return only the first result of the query. Returns all otherwise.
        -and_query: bool, if True will query records with all parameters true. If False, an OR query is performed.


    """
    query = db.query(UserRelationship).filter(
        (UserRelationship.user_id.like(user_id)) & (UserRelationship.friend_id.like(friend_id))
    )

    return query.first() if first else query.all()

def delete_relationship(db: Session, user_id: str="%", friend_id: str="%", first: bool=True):
    """
    Queries a database using a Session object given a user_id and friend_id, deleting the corresponding record.
    """    

    query = db.query(UserRelationship).filter((UserRelationship.user_id==user_id) & (UserRelationship.friend_id==friend_id)).delete()
    db.commit()