from __future__ import annotations
from sqlalchemy import Integer, Column, String, Boolean, Enum, UniqueConstraint, ForeignKey, DateTime, Float, Interval
from sqlalchemy.orm import relationship
import enum
from database import Base

class RelationshipType(enum.Enum):
    PENDING = 0
    ACCEPTED = 1
    BLOCKED = 2

class UserRelationship(Base):
    __tablename__ = "UserRelationships"

    user_id = Column(Integer, ForeignKey("users.id"),primary_key=True)
    friend_id = Column(Integer,primary_key=True)
    relationship_status = Column(Enum(RelationshipType))

    # ensures each user->friend record can only occur once
    UniqueConstraint("user_id", "friend_id", name="unique_friendship")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(20))
    email = Column(String(30), unique=True, index=True)
    hashed_pass = Column(String(30))
    disabled = Column(Boolean)

    """
    form connection to userrelationship table, with cascade mode ensuring that when the elements of the relationships array are deleted
    the corresponding db record is deleted too. 
    the lazy parameter loads the children as queries rather than objects, allowing filters to be performed on the relationships array
    """ 
    relationships = relationship("UserRelationship",cascade="all, delete, delete-orphan", lazy="dynamic")
    recipes = relationship("Recipe", cascade="all, delete, delete-orphan", lazy="dynamic")

class Recipe(Base):
    __tablename__ = "recipes"

    id = Column(Integer, primary_key=True)
    name = Column(String(2000))
    description = Column(String(7000))
    servings = Column(String(2000))
    prep_time = Column(Interval)
    total_time = Column(Interval)
    date_published = Column(DateTime)
    category = Column(String(2000))
    aggregated_rating = Column(Float)
    review_count = Column(Float)
    calories = Column(Float)
    fat = Column(Float)
    saturated_fat = Column(Float)
    cholesterol = Column(Float)
    sodium = Column(Float)
    carbohydrate = Column(Float)
    fiber = Column(Float)
    sugar = Column(Float)
    protein = Column(Float)
    servings = Column(Float)
    recipe_yield = Column(String(2000))
        
    author = Column(Integer, ForeignKey("users.id"))
    
    steps = relationship("RecipeStep", cascade="all, delete, delete-orphan", lazy="dynamic")
    ingredients = relationship("RecipeIngredient", cascade="all, delete, delete-orphan", lazy="dynamic")
    images = relationship("RecipeImage", cascade="all, delete, delete-orphan", lazy="dynamic")
    keywords = relationship("RecipeKeyword", cascade="all, delete, delete-orphan", lazy="dynamic")

class RecipeStep(Base):
    __tablename__ = "recipesteps"
    id = Column(Integer, primary_key=True)
    recipe = Column(Integer, ForeignKey("recipes.id"))
    text = Column(String(2000))
    index = Column(Integer)
    # ensures each recipe / index pair is unique
    UniqueConstraint("recipe", "index", name="unique_recipe_item")

class RecipeIngredient(Base):
    __tablename__ = "recipeingredients"
    id = Column(Integer, primary_key=True)
    recipe = Column(Integer, ForeignKey("recipes.id"))
    text = Column(String(2000))
    index = Column(Integer)
    quantity = Column(String(2000))

    # ensures each recipe / index pair is unique
    UniqueConstraint("recipe", "index", name="unique_recipe_ingredient_item")

class RecipeImage(Base):
    __tablename__ = "recipeimages"
    id = Column(Integer, primary_key=True)
    recipe = Column(Integer, ForeignKey("recipes.id"))
    url = Column(String(2000))
    index = Column(Integer)
    
    # ensures each recipe / index pair is unique
    UniqueConstraint("recipe", "index", name="unique_recipe_image_url")

class RecipeKeyword(Base):
    __tablename__ = "recipekeywords"
    id = Column(Integer, primary_key=True)
    recipe = Column(Integer, ForeignKey("recipes.id"))
    keyword = Column(String(2000))
    index = Column(Integer)

