import logging
from typing import Dict, List

import pandas as pd

from recipe_rec.utilities import get_recipes

RANDOM_STATE: int = 42

logging.basicConfig(level=logging.INFO)

recipes: pd.DataFrame = get_recipes("./data/recipes.csv")

ingredient_indexes: Dict[str, int] = {}
unique_ingredients: List[str] = []

# get matrix size
for recipe in recipes["RecipeIngredientParts"]:

    for ingredient in recipe:

        if ingredient not in ingredient_indexes.keys():

            ingredient_indexes[ingredient] = len(unique_ingredients)
            unique_ingredients.append(ingredient)
