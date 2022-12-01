from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, HTTPException, status
import crud
from passlib.context import CryptContext
from typing import Union
from schemas import UserInDB, TokenData
from database import SessionLocal
from datetime import datetime, timedelta
from jose import JWTError, jwt
from config import SECRET_KEY, ALGORITHM

# hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Compares a string password with a hash.

    Inputs:
        - `plain_password: str`, the plaintext password
        - `hashed_password: str`, the hashed password

    Outputs:
        - `bool`, `True` if the plaintext matches the hash and `False` otherwise
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Applies the Bcrypt hashing algorithm to a plaintext password.

    Inputs:
        - `password: str`, the plaintext password to hash

    Outputs:
        - `str`, the hashed password
    """
    return pwd_context.hash(password)

def authenticate_user(db, username: str, password: str) -> Union[bool, UserInDB]:
    """
    Verifies if a username and password combination occur in a given database.

    Inputs:
        - `username: str` - the username to query the database with
        - `password: str` - the password to hash and query the database with
    
    Outputs:
        - `Union[bool, DBUser]` - returns `False` if the username is not in the database or if the username does occur but the password is incorrect. Returns the `user` object otherwise.
    """

    user = crud.get_users_and(db, username=username, first=True)

    if not user:
        return False
    if not verify_password(password, user.hashed_pass):
        return False
    return user
    

def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=15)) -> str:
    """
    Creates an access token with some data as the subject.

    Inputs:
        - `data: dict`, the data to create the access token with. Requires a value with the key `"sub"`.
        - `expires_delta: timedelta`, an amount of time to allow the token to be usable; it will not be valid after the given duration expires.

    Outputs:
        - `str`, the string representation of the JWT access token
    """

    # ensure a copy is made to avoid changing original
    to_encode = data.copy()

    # calculate expiry time
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})

    # generare access token with JWT
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# oauth2 scheme will extract token from header
async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Gets the current user given a JWT token. Dependency on `oauth2_scheme` will retrieve the JWT token from the Authorization header of an HTTP request.

    Inputs:
        - `token: str`, the access token for the user

    Outputs:
        - `UserInDB`, the object for the current user as determined by the access token
    """
    # create exception
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # use token and secret key to decode
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        # create token object using username and token
        email: str = payload.get("sub")

        # if no email, the token is invalid
        if email is None:
            raise credentials_exception

        # extract token data
        token_data = TokenData(username=email)
    except JWTError:
        raise credentials_exception

    """
    user = user_db.get(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user
    """
    
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """
    Checks the disabled attribute of a user, returning the same `User` object if the not disabled; an `HTTPException` is thrown otherwise.

    Inputs:
        - `current_user: User`, the user to to check

    Outputs:
        - User, the user object - if the disabled attribute is `False`
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
