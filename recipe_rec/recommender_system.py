import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from recipe_rec.data import store
from recipe_rec.utilities import get_dataset


def build_timer(f):
    """
    Decorator used to measure the time to build a recommender system.
    """

    def inner(self, *args, **kwargs):

        start: int = time.time_ns()

        f(self, *args, **kwargs)

        end: int = time.time_ns()

        self.build_time: int = end - start

    return inner


def rec_timer(f):
    """
    Decorator used to measure the time for a system to produce a recommendation and
    updates the object's store of all recommendation execution times, re-calculating the average.
    """

    def inner(self, *args, **kwargs):

        start: int = time.time_ns()

        ret: Any = f(self, *args, **kwargs)

        end: int = time.time_ns()

        self.rec_times["times"].append(end - start)
        self.rec_times["avg"] = sum(self.rec_times["times"]) / len(
            self.rec_times["times"]
        )

        return ret

    return inner


class RecommenderSystem:
    """
    A base class for recipe recommender systems.
    """

    # recipes = store["recipes"]
    # unique_ingredients = store["unique_ingredients"]

    def __init__(self) -> None:

        # common attributes among all systems
        self.recipes = get_dataset()

        # unique ID for the class' instantiation
        self.execution_id: str = str(uuid.uuid4().hex)
        # filepaths of associated disk data
        self.disk_data: Dict[str, Path] = {}

        self.rec_times: Dict[str, Union[List[int], float]] = {"times": [], "avg": 0.0}

    def build_ingredient_index(self, num_trees: int, out_path: Path) -> None:
        """
        Builds an `AnnoyIndex` of ingredients, allowing for easy ingredient recommendation.

        Parameters:
            - `num_trees: int`: the number of trees to use when building the `AnnoyIndex`.
            - `out_path: pathlib.Path`: the filepath to write the `AnnoyIndex` to.

        """

        ingredient_embeddings: List[np.array] = []
        # get vectors for each ingredient using recipe vectorizer
        for ingredient in self.unique_ingredients:

            ingredient_embed: np.array = self.recipe_vectorizer([ingredient])

            ingredient_embeddings.append(ingredient_embed)

        self.build_index(
            iterable=ingredient_embeddings,
            num_trees=num_trees,
            out_path=out_path,
            recipe_index=False,
        )

    def build_index(
        self,
        iterable: Iterable,
        num_trees: int,
        out_path: Union[Path, None],
        recipe_index: bool = True,
    ) -> None:
        """
        Builds an `AnnoyIndex`, populating it using an iterable of vectors. The index of each vector in the iterable is used
        as its index in the index. The resulting index is written to disk if `out_path` is not `None`.

        Parameters:
            - `iterable: Iterable`: an iterable containing vectors to write into the `AnnoyIndex`.
            - `num_trees: int`: the number of trees to use when constructing the `AnnoyIndex`.
            - `out_path: Union[Path, None]`: the filepath to write the `AnnoyIndex` to. If `None`, the index is not written to disk.
            - `recipe_index: bool = True`: if `True`, the index is set as the class' `recipe_index` attribute. Otherwise, the index is an
        ingredient index and is set as the class' `ingredient_index` attribute`.
        """

        # create index using class attributes
        index: AnnoyIndex = AnnoyIndex(self.vec_size, self.index_distance_metric)

        # populate index
        for i, v in enumerate(iterable):
            index.add_item(i, v)

        # build and save
        index.build(num_trees)

        if out_path is not None:

            # convert to str; annoy doesn't support pathlib
            out_path_str = out_path.absolute().as_posix()
            index.save(out_path_str)

        # set to relevant class attribute
        if recipe_index:
            self.recipe_index: AnnoyIndex = index
        else:
            self.ingredient_index: AnnoyIndex = index

    def load_index(self, index_path: Path, recipe_index: bool = True) -> None:
        """
        Loads an `AnnoyIndex` from disk and sets it as the appropriate class attribute.

        Parameters:
            - `index_path: pathlib.Path`: the path to load the `AnnoyIndex` from.
            - `recipe_index: bool = True`: if True, the index is stored at the `recipe_index` of the class. Otherwise, the index is
        stored at the `ingredient_index` attribute.
        """

        index_path_str = index_path.absolute().as_posix()

        index: AnnoyIndex = AnnoyIndex(self.vec_size, self.index_distance_metric)
        index.load(index_path_str)

        if recipe_index:
            self.recipe_index = index
        else:
            self.ingredient_index = index

    @rec_timer
    def get_recommendations(
        self,
        recipe: List[str],
        n_recommendations: int = 10,
        search_id: int = None,
        get_recipes: bool = True,
    ) -> Union[pd.DataFrame, List[str]]:
        """
        Creates a recipe vector from a list of ingredients and queries the Annoy index for the `n_recommendations` nearest neighbours.
        Raises a `KeyError` if the recipe cannot be vectorized.

        Parameters
            - `recipe`: `List[str]`: a list of string ingredients
            - `n_recommendations`: `int = 10`: the number of recommendations to return
            - `search_id: int = None`: the index of the querying recipe. If not `None`, this recipe will not be returned as a recommendation.
            - `get_recipes: bool = True`: if True, recommends recipes; if False, reccommends ingredients
        Returns:
            - `Union[pd.DataFrame, List[str]`, a sorted DataFrame of the recommended recipes, or a list of recommended ingredients.
        """

        try:

            # get the vector of the recipe
            recipe_vec: np.array = self.recipe_vectorizer(recipe)

            if get_recipes:
                # get closest vectors from the dataset
                rec_indexes: List[int] = self.recipe_index.get_nns_by_vector(
                    recipe_vec, n_recommendations
                )

            else:
                # get closest vectors from the ingredients dataset
                rec_indexes: List[int] = self.ingredient_index.get_nns_by_vector(
                    recipe_vec, n_recommendations
                )

            # if there is a search id
            if search_id is not None:

                # if the search is in the results
                if search_id in rec_indexes:

                    if get_recipes:
                        # get another recommenation (shouldn't be the same one again)
                        rec_indexes = self.recipe_index.get_nns_by_vector(
                            recipe_vec, n_recommendations + 1
                        )
                    else:
                        # get another recommenation (shouldn't be the same one again)
                        rec_indexes = self.ingredient_index.get_nns_by_vector(
                            recipe_vec, n_recommendations + 1
                        )
                    # filter out the search query
                    rec_indexes = [rec for rec in rec_indexes if rec != search_id]

            if get_recipes:
                # map recommendations to recipes
                recs: pd.DataFrame = self.recipes.iloc[rec_indexes].copy()
            else:
                # map recommendations to ingredients
                recs: List[str] = [self.unique_ingredients[i] for i in rec_indexes]

            return recs

        except KeyError:
            raise ValueError(
                "One of the given ingredients did not exist in the training dataset."
            )
