import pandas as pd

from recipe_rec.data_loader import get_recipes

RANDOM_STATE = 42
recipes: pd.DataFrame = get_recipes("./data/recipes.csv")
