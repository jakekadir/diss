from typing import List, NoReturn

import pandas as pd

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem, build_timer, rec_timer


class SimpleSearch(RecommenderSystem):
    @build_timer
    def __init__(self) -> None:

        super().__init__()

    # overwrite parent method
    @rec_timer
    def get_recommendations(
        self, recipe: List[str], n_recommendations: int = 10, search_id: int = None
    ) -> pd.DataFrame:

        # count length of the intersection of the set of the query ingredients and set of recipe ingredients
        counts: pd.Series = recipes["RecipeIngredientParts"].apply(
            lambda ingrs: len(set(ingrs).intersection(set(recipe)))
        )

        # sort for largest intersection
        counts = counts.sort_values(ascending=False)

        rec_indexes: List[int] = list(counts.head(n_recommendations).index)

        # if there is a search id
        if search_id is not None:

            # if the search is in the results
            if search_id in rec_indexes:

                # get another recommenation (shouldn't be the same one again)
                rec_indexes = list(counts.head(n_recommendations + 1).index)

                # filter out the search query
                rec_indexes = [rec for rec in rec_indexes if rec != search_id]

        # get top n_recommendations
        return recipes.iloc[rec_indexes].copy()

    # overwrite irrelevant methods
    def build_ingredient_index(self) -> NoReturn:
        raise NotImplementedError

    def build_index(
        self,
        iterable,
        num_trees: int,
        out_path: str,
        recipe_index: bool = True,
        save: bool = True,
    ) -> NoReturn:
        raise NotImplementedError

    def load_index(self, index_path: str) -> NoReturn:
        raise NotImplementedError
