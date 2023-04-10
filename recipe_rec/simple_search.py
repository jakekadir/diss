from typing import List

import pandas as pd

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem


class SimpleSearch(RecommenderSystem):
    def get_recommendations(
        self, ingredients: List[str], n_recommendations: int
    ) -> pd.DataFrame:

        # list of lists of ingredients
        recipe_ingredients = recipes["RecipeIngredientParts"]

        # count length of the intersection of the set of the query ingredients and set of recipe ingredients
        counts = recipe_ingredients.apply(
            lambda ingrs: len(list(set(ingrs).intersection(set(ingredients))))
        )

        # sort for largest intersection
        counts = counts.sort_values(ascending=False)

        # get top n_recommendations
        return recipes.iloc[counts.head(n_recommendations).index]
