import ast

import pandas as pd


# function to parse strings of lists as Python lists
def parseTupleFunc(tuple_str: str):

    try:
        return ast.literal_eval(tuple_str)

    except Exception as e:

        print(tuple_str)


def recipe_loader():

    # import column
    recipes = pd.read_csv("../data/recipes.csv")

    recipes = recipes.drop(
        recipes[recipes["RecipeIngredientParts"].str[:2] != "c("].index
    )

    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].str[1:]

    recipes["Ingredients"] = recipes["RecipeIngredientParts"].apply(parseTupleFunc)

    return recipes
