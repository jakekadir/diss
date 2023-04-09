from pathlib import Path
from typing import List
import pandas as pd


class RecommenderSystem:
    """
    A base class for recommender systems.
    """

    def __init__(self, name: str, path: Path, vec_size: int):

        self.name = name
        self.index_path = path
        self.vec_size = vec_size


class IngredientRecommender(RecommenderSystem):
    """
    A base class for recommender systems that recommend using ingredients.
    """

    def get_recommendations(
        self, recipe: List[str], n_recommendations: int
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

            print(rec_indexes)
            # translate recommendations into recipes
            recs = self.recipes.iloc[rec_indexes]

            return recs

        except KeyError:
            raise ValueError(
                "One of the given ingredients did not exist in the training dataset."
            )
