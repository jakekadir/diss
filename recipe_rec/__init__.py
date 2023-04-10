import pandas as pd

from recipe_rec.data_loader import get_recipes

recipes: pd.DataFrame = get_recipes("./data/recipes.csv")
