from typing import List

import pandas as pd

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem


class SimpleSearch(RecommenderSystem):
    def get_recommendations(
        self, recipe: List[str], n_recommendations: int = 10, search_id: int = None
    ) -> pd.DataFrame:
        
        super().__init__()

        # list of lists of ingredients
        recipe_ingredients = recipes["RecipeIngredientParts"]

        # count length of the intersection of the set of the query ingredients and set of recipe ingredients
        counts = recipe_ingredients.apply(
            lambda ingrs: len(list(set(ingrs).intersection(set(recipe))))
        )

        # sort for largest intersection
        counts = counts.sort_values(ascending=False)

        rec_indexes = counts.head(n_recommendations).index

        # if there is a search id
        if search_id is not None:

            # if the search is in the results
            if search_id in rec_indexes:

                # get another recommenation (shouldn't be the same one again)
                rec_indexes = counts.head(n_recommendations + 1).index

                # filter out the search query
                rec_indexes = [rec for rec in rec_indexes if rec != search_id]

        # get top n_recommendations
        return recipes.iloc[rec_indexes].copy()
