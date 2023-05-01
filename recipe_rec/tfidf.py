import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from recipe_rec.data import store
from recipe_rec.recommender_system import RecommenderSystem, build_timer, rec_timer
from recipe_rec.utilities import (
    check_dataset_loaded,
    check_file_exists,
    check_is_dir,
    space_replacer,
)


class TfIdfRecommender(RecommenderSystem):
    """
    A recommender system using pre-trained TF-IDF vectors. Creates a TF-IDF vectorizer for the recipe's corpus and uses
    scikit-learn's `NearestNeighbors` algorithm to find nearest neighbors.

    Parameters:
        - `vectorizer_path: pathlib.Path = None` (optional): the path of pre-trained TF-IDF vectorizer.
        - `model_path: Path = None` (optional): the path of a pre-built `NearestNeighbors` instance.
        - `output_dir: pathlib.Path = pathlib.Path(".")` (optional): the base directory to use when writing files to disk.
        - `verbose: bool = True` (optional): outputs updates during training if True, outputs nothing otherwise.
    """

    @build_timer
    def __init__(
        self,
        vectorizer_path: Path = None,
        model_path: Path = None,
        output_dir: Path = Path("."),
        verbose: bool = False,
    ) -> None:

        super().__init__()

        # constants
        self.output_dir: Path = check_is_dir(output_dir)
        self.verbose: bool = verbose

        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL)

        self.disk_data: Dict[str, Path] = {
            "vectorizer_path": vectorizer_path,
            "model_path": model_path,
        }
        # validate provided file paths before trying to do anything
        for filepath in self.disk_data.values():
            if filepath is not None:
                check_file_exists(filepath)

        # if no model
        if self.disk_data["vectorizer_path"] is None:

            # train a model
            self.generate_vectors()
        # otherwise
        else:
            # load the model
            with open(self.disk_data["vectorizer_path"], "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

        if self.disk_data["model_path"] is None:

            self.build_nearest_neighbors()

        else:

            with open(self.disk_data["model_path"], "rb") as f:
                self.model = pickle.load(f)

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients in a recipe to the TF-IDF vector of the concatenated string of these ingredients.

        Parameters:
        `recipe: List[str]`: a list of ingredient tokens which comprise the recipe.

        Returns:
        `np.array`: the recipe's vector.
        """

        # replace spaces in the query
        recipe = space_replacer(recipe)

        # combine ingredients into a string
        joined_ingredients: str = ",".join(recipe)

        # retrieve TF-IDF vector for string
        recipe_vec: csr_matrix = self.tfidf_vectorizer.transform([joined_ingredients])

        recipe_vec = recipe_vec.toarray()

        return recipe_vec

    def generate_vectors(self):
        """
        Generates TF-IDF vectors for the entire recipe dataset, stored at the `ingredient_vectors` attribute of the class.
        """

        preprocessed_ingredients = self.recipes["RecipeIngredientParts"].apply(
            space_replacer
        )
        preprocessed_ingredients = preprocessed_ingredients.str.join(" ")

        self.tfidf_vectorizer = TfidfVectorizer()

        # generate vectors; stored as a sparse matrix
        self.ingredient_vectors: csr_matrix = self.tfidf_vectorizer.fit_transform(
            preprocessed_ingredients
        )

        vectorizer_path: Path = Path(
            self.output_dir, f"tfidf_vectorizer_{self.execution_id}.pkl"
        )

        with open(vectorizer_path, "wb") as f:

            pickle.dump(self.tfidf_vectorizer, f)

        return vectorizer_path

    def build_nearest_neighbors(self) -> Path:

        self.model = NearestNeighbors(n_neighbors=10, metric="manhattan")
        self.model.fit(self.ingredient_vectors)

        model_path: Path = Path(
            self.output_dir, f"nearest_neighbours_{self.execution_id}.pkl"
        )

        with open(model_path, "wb") as f:

            pickle.dump(self.model, f)

        return model_path

    @rec_timer
    def get_recommendations(
        self, recipe: List[str], n_recommendations: int = 10, search_id: int = None
    ) -> pd.DataFrame:
        """
        Creates a recipe vector from a list of ingredients and performs a nearest neighbor search using scikit-learn `n_recommendations` nearest neighbours.
        Raises a `KeyError` if the recipe cannot be vectorized.

        Parameters
            - `recipe`: `List[str]`: a list of string ingredients
            - `n_recommendations`: `int = 10`: the number of recommendations to return
            - `search_id: int = None`: the index of the querying recipe. If not `None`, this recipe will not be returned as a recommendation.
        Returns:
            - `pd.DataFrame`, a sorted DataFrame of the recommended recipes.
        """

        recipe_vec = self.recipe_vectorizer(recipe)

        _, rec_indexes = self.model.kneighbors(
            recipe_vec, n_neighbors=n_recommendations
        )

        # get the first row of recommendations
        rec_indexes = rec_indexes[0]

        # if there is a search id
        if search_id is not None:

            # if the search is in the results
            if search_id in rec_indexes:

                # get another recommenation (shouldn't be the same one again)
                _, rec_indexes = self.model.kneighbors(
                    recipe_vec, n_neighbors=n_recommendations + 1
                )
                rec_indexes = rec_indexes[0]

                # filter out the search query
                rec_indexes = [rec for rec in rec_indexes if rec != search_id]

        # map recommendations to recipes
        recs: pd.DataFrame = self.recipes.iloc[rec_indexes].copy()

        return recs
