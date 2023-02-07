from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, status, Depends, Form
import crud
from dependencies import get_db
from models import Recipe, RecipeImage, RecipeIngredient, RecipeKeyword, RecipeStep
from datetime import timedelta
import re

import ast
import pandas as pd

router = APIRouter()

def time_parser(time_str: str) -> timedelta:
    
    # regex str
    time_extraction: str = "(([0-9]+)H)?(([0-9]+)M)?"
    
    # perform search
    search = re.search(time_extraction, time_str)
    
    # if the pattern matches
    if search:
        
        # create time object with group 2 as hour and group 4 as minute
        hours = int(search.group(2)) if search.group(2) is not None else 0
        minutes = int(search.group(4)) if search.group(4) is not None else 0
        
        # if more than 24 hours, convert to days
        days = hours // 24
        hours -= days * 24
        
        time_val = timedelta(days, hours, minutes, 0)
        return time_val
    
    else:
        return None
    

def load_data():
    # load dataframe
    recipes = pd.read_csv("../data/recipes.csv")

    # parse columns with multiple values per column
    multi_cols = [
        "Images",
        "RecipeInstructions",
        "RecipeIngredientParts",
        "RecipeIngredientQuantities",
        "Keywords"
    ]
    
    for col in multi_cols:

        # drop malformed rows
        recipes = recipes.drop(recipes[recipes[col].str[:2] != "c("].index)

        # strip the leading c from relevant rows
        recipes[col] = recipes[col].str[1:]
        
        # replace NA values with empty string
        regs = ["(\()(NA)(,)","(, )(NA)(,)", "(, )(NA)(\))"]
        for reg in regs:
            recipes["RecipeIngredientQuantities"] = recipes["RecipeIngredientQuantities"].str.replace(reg, r'\1""\3', regex=True)

    # fill NaNs
    str_cols = recipes.select_dtypes(include=['object']).columns
    recipes.loc[:, str_cols] = recipes.loc[:, str_cols].fillna("")

    float_cols = recipes.select_dtypes(include=['float64']).columns
    recipes.loc[:, float_cols] = recipes.loc[:, float_cols].fillna(0.0)

    int_cols = recipes.select_dtypes(include=['int64']).columns
    recipes.loc[:, int_cols] = recipes.loc[:, int_cols].fillna(0)
    
    # correctly format times
    time_cols = ["PrepTime", "CookTime", "TotalTime"]
    
    for time_col in time_cols:
        # strip "PT" prefix
        recipes[time_col] = recipes[time_col].str.slice(start=2)
        recipes[time_col] = recipes[time_col].apply(time_parser)
        
    # format dates
    recipes["DatePublished"] = pd.to_datetime(recipes["DatePublished"])

    return recipes


def parseTupleFunc(tupleStr):
    
    parsed_tuple = []

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

    def saveRecipe(recipe):

        # create recipe record
        recipe_dict = {
            "name" : recipe["Name"],
            "description" : recipe["Description"],
            "servings" : recipe["RecipeServings"],
            "prep_time" : recipe["PrepTime"],
            "total_time" : recipe["TotalTime"],
            "date_published" : recipe["DatePublished"],
            "category" : recipe["RecipeCategory"],
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

        recipe_record = Recipe(**recipe_dict)

        # create image record
        image_urls = parseTupleFunc(recipe["Images"])
        for i in range(len(image_urls)):

            image = RecipeImage(**{"url" : image_urls[i], "index" : i})

            recipe_record.images.append(image)

        # create steps record
        steps = parseTupleFunc(recipe["RecipeInstructions"])
        for i in range(len(steps)):

            step = RecipeStep(**{"text" : steps[i], "index" : i})

            recipe_record.steps.append(step)

        # create ingredients record
        ingredients = parseTupleFunc(recipe["RecipeIngredientParts"])
        ingredient_quantities = parseTupleFunc(recipe["RecipeIngredientQuantities"])
        for i, (ingredient, quantity) in enumerate(zip(ingredients, ingredient_quantities)):

            ingredient = RecipeIngredient(**{"text" : ingredient, "index" : i, "quantity" : quantity})

            recipe_record.ingredients.append(ingredient)

        # create keywords record
        keywords = parseTupleFunc(recipe["Keywords"])
        for i in range(len(keywords)):

            keyword = RecipeKeyword(**{"keyword" : keywords[i], "index" : i})

            recipe_record.keywords.append(keyword)
            
        db.add(recipe_record)

    print("LOADING DATA")
    recipes = load_data()
    
    print("BEGINNING TO CREATE RECIPE RECORDS")
    print("TOTAL RECIPES: ", len(recipes))
    for i in range(69,(len(recipes) // 1000) + 1):
        print(f"RECIPES {i * 1000} TO {(i + 1 ) * 1000 - 1}")
        recipes.iloc[i * 1000: ((i+1) * 1000) - 1].apply(saveRecipe, axis=1)
        db.commit()
        print(f"COMITTED RECIPES {i * 1000} TO {(i + 1 ) * 1000 - 1}\n")
    # recipes.apply(saveRecipe, axis=1)
    
    print("ABOUT TO COMMIT DB")

    
    print("COMMITTED DB, COMPLETE")