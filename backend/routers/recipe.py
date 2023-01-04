from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, status, Depends, Form
import crud
from dependencies import get_db
from models import Recipe, RecipeImage, RecipeIngredient, RecipeKeyword, RecipeStep

import ast
import pandas as pd

router = APIRouter()
# load dataframe
recipes = pd.read_csv("../data/recipes.csv")

def parseTupleFunc(tupleStr):

    try:
        return ast.literal_eval(tupleStr)

    except Exception as e:

        print(tupleStr)

@router.post("/populate-recipes", status_code=201)
async def populate_recipes(db: Session = Depends(get_db)):
    """
    Keep the columns:
    -Name
    -AuthorId
    PrepTime
    TotalTime
    DatePublished
    Description
    Images (multiple)
    RecipeCategory
    Keywords (multiple)
    RecipeIngredientQuantities (multipled, used for ingredient table)
    Aggregated Rating
    Review Count
    Calories
    FatContent
    SaturedFatContent
    CholesterolContent
    SodiumContent
    CarbohydrateContent
    FiberContent
    SugarContent
    ProteinContent
    RecipeServings
    RecipeYield
    RecipeInstructions (multiple)
    """

    # for each row
    for recipe in recipes[:5].iterrows():

        print(recipe)
        # create recipe record
        recipe_dict = {
            "description" : recipe["Description"],
            "servings" : recipe["Servings"],
            "prep_time" : recipe["PrepTime"],
            "total_time" : recipe["TotalTime"],
            "date_published" : recipe["DatePublished"],
            "category" : recipe["Category"],
            "aggregated_rating" : recipe["AggregatedRating"],
            "review_count" : recipe["ReviewCount"],
            "calories" : recipe["Calories"],
            "fat" : recipe["FatContent"],
            "saturated_fat" : recipe["SaturatedFatContent"],
            "cholesterol" : recipe["CholesterolContent"],
            "sodium" : recipe["SodiumContent"],
            "carbohydrate" : recipe["CarbohydrateContent"],
            "fiber" : recipe["FiberContent"],
            "sugar": recipe["SugarContent"],
            "protein" : recipe["ProteinContent"],
            "servings" : recipe["RecipeServings"],
            "recipe_yield" : recipe["RecipeYield"]
        }

        recipe = Recipe(recipe_dict)

        # create image record
        image_urls = parseTupleFunc(recipe["Images"])
        for i in range(len(image_urls)):

            image = RecipeImage({"url" : image_urls[i], "index" : i})

            db.images.append(image)

        # create steps record
        steps = parseTupleFunc(recipe["RecipeInstructions"])
        for i in range(len(steps)):

            step = RecipeStep({"text" : steps[i], "index" : i})

            db.images.append(step)

        # create ingredients record
        ingredients = parseTupleFunc(recipe["RecipeIngredients"])
        for i in range(len(ingredients)):

            ingredient = RecipeIngredient({"text" : ingredients[i], "index" : i})

            db.ingredients.append(ingredient)

        # create keywords record
        keywords = parseTupleFunc(recipe["RecipeKeyword"])
        for i in range(len(keywords)):

            keyword = RecipeKeyword({"text" : keywords[i], "index" : i})

            db.keywords.append(keyword)

        db.add(recipe)
    db.commit()
    db.refresh(recipe)