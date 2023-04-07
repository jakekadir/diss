import pandas as pd
import ast
import pathlib
from typing import List

def parseTupleFunc(tupleStr: str):

    try:
        return ast.literal_eval(tupleStr)

    except Exception as e:

        print(tupleStr)

def get_recipes(path: pathlib.Path) -> pd.DataFrame:
    """
    Opens the recipe dataset at the specified path, converting the nested fields to array-like objects rather than strings.

    Inputs:
        path: pathlib.Path, the path to the recipe dataset
    Outputs:
        pd.DataFrame, the prepared DataFrame of recipes
    """
    
    # ingest data
    recipes: pd.DataFrame = pd.read_csv(path)

    # embedded array columns
    list_column_names: List[str] = ["RecipeInstructions", "RecipeIngredientParts", "Keywords", "Images"] # plus, needs tweaking "RecipeIngredientQuantities" 
    
    for col in list_column_names:
            
        recipes = recipes.drop(recipes[recipes[col].str[:2] != "c("].index)

        recipes[col] = recipes[col].str[1:]

        parseTuple = lambda tupleStr: ast.literal_eval(tupleStr)

        recipes[col] = recipes[col].apply(parseTupleFunc)

    return recipes

