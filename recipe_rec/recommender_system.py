import uuid
from pathlib import Path
from typing import List

import pandas as pd

from recipe_rec import recipes


class RecommenderSystem:
    """
    A base class for recommender systems.
    """

    def __init__(self):

        # common attributes among all systems

        # unique ID for the class' instantiation
        self.execution_id = str(uuid.uuid4().hex)
        # filepaths of associated disk data
        self.disk_data = {}


class IngredientRecommender(RecommenderSystem):
    """
    A base class for recommender systems that recommend using ingredients.
    """

    def __init__(self):
        super().__init__()

    def get_recommendations(
        self, recipe: List[str], n_recommendations: int = 10, search_id: int = None
    ) -> pd.DataFrame:
        """
        Creates a recipe vector from a list of ingredients and queries the Annoy index for the `n_recommendations` nearest neighbours.

        Inputs:
            - `recipe`: `List[str]`, a list of string ingredients
            - `n_recommendations`: `int`, the number of recommendations to return
        Outputs:
            - `pd.DataFrame`, a sorted DataFrame of the recommended recipes
        """

        try:

            # get the vector of the recipe
            recipe_vec = self.recipe_vectorizer(recipe)

            # get closest vectors from the dataset
            rec_indexes = self.index.get_nns_by_vector(recipe_vec, n_recommendations)

            # if there is a search id
            if search_id is not None:

                # if the search is in the results
                if search_id in rec_indexes:

                    # get another recommenation (shouldn't be the same one again)
                    rec_indexes = self.index.get_nns_by_vector(
                        recipe_vec, n_recommendations + 1
                    )

                    # filter out the search query
                    rec_indexes = [rec for rec in rec_indexes if rec != search_id]

            # map recommendations to recipes
            recs = recipes.iloc[rec_indexes].copy()

            return recs

        except KeyError:
            raise ValueError(
                "One of the given ingredients did not exist in the training dataset."
            )
