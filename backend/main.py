import config
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from dotenv import dotenv_values
from pymongo import MongoClient
from routes import router as user_router

config = dotenv_values(".env")

app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.on_event("startup")
def startup_db_client():
    app.mongodb_client = MongoClient(config["MONGO_URI"])
    app.db = app.mongodb_client[config["DB_NAME"]]

    print("Connected to CosmoDB database.")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()

app.include_router(user_router, tags=["users"], prefix="/user")

@app.get("/")
async def get_root():
    return {"message" : "hello"}

@app.get("/hidden/")
async def get_hidden(token: str = Depends(oauth2_scheme)):
    return {"token" : token}







"""
from flask import Flask
mongoengine.connect(host=config.MONGO_URI)

app = Flask(__name__)
app.config["MONGO_URI"] = config.MONGO_URI

# client = MongoClient(config.MONGO_URI)

# print(client.db.food)

@app.route("/")
def hello() -> None:
    return "<h1>Hello World</h1>"

@app.route("/login")
def login():
    return "<h1>Login Page</h1>"

@app.route("/register")
def register():
    return "<h1>Register Page</h1>"
"""