from recommender_system import IngredientRecommender
from data_loader import get_recipes
from typing import List
import pandas as pd


class SimpleSearch(IngredientRecommender):
    def __init__(self):

        super().__init__("", "", "")

        self.recipes = get_recipes("../data/recipes.csv").reset_index()

    def recipe_search(
        self, ingredients: List[str], n_recommendations: int
    ) -> pd.DataFrame:

        # list of lists of ingredients
        recipe_ingredients = self.recipes["RecipeIngredientParts"]

        # count length of the intersection of the set of the query ingredients and set of recipe ingredients
        counts = recipe_ingredients.apply(
            lambda ingrs: len(list(set(ingrs).intersection(set(ingredients))))
        )

        # sort for largest intersection
        counts = counts.sort_values(ascending=True)

        # get top n_recommendations
        return self.recipes.iloc[counts.head(n_recommendations).index]
