from pymongo import MongoClient
from pymongo.database import Database
from config import MONGO_URI, DB_NAME

# connect to cluster
client: MongoClient = MongoClient(MONGO_URI)

db: Database = client[DB_NAME]