import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

store = {}


def parse_string_as_tuple(tuple_string: str) -> List[str]:
    """
    Takes a strings and safely attempts to parse it as Python code.
    """

    try:

        parsed: Tuple[str] = ast.literal_eval(tuple_string)

        # was not parsed as desired, so reject
        if type(parsed) is not tuple:

            return tuple_string

        else:

            return parsed

    except Exception:

        return tuple_string


def get_recipes(path: Path) -> pd.DataFrame:
    """
    Opens the recipe dataset at the specified path, converting the nested fields to array-like objects rather than strings.

    Inputs:
        path: pathlib.Path, the path to the recipe dataset
    Outputs:
        pd.DataFrame, the prepared DataFrame of recipes
    """

    # ingest data
    recipes: pd.DataFrame = pd.read_csv(path)

    # format the ingredients
    recipes = recipes.drop(
        recipes[recipes["RecipeIngredientParts"].str[:2] != "c("].index
    )
    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].str[1:]

    # parse the string as a tuple
    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].apply(
        parse_string_as_tuple
    )

    # reset index for consistency
    recipes = recipes.reset_index(drop=True)

    return recipes


def load_and_set_data(path: Path, store=store):
    recipes: pd.DataFrame = get_recipes(path)

    ingredient_indexes: Dict[str, int] = {}
    unique_ingredients: List[str] = []

    # get matrix size
    for recipe in recipes["RecipeIngredientParts"]:

        for ingredient in recipe:

            if ingredient not in ingredient_indexes.keys():

                ingredient_indexes[ingredient] = len(unique_ingredients)
                unique_ingredients.append(ingredient)

    store["recipes"] = recipes
    store["unique_ingredients"] = unique_ingredients


def load_dataset(path: Path):

    load_and_set_data(path)
