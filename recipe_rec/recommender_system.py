import time
import uuid
from pathlib import Path
from typing import List
from annoy import AnnoyIndex

import pandas as pd

from recipe_rec import recipes, unique_ingredients


def build_timer(f):
    """
    Decorator used to measure the time to build a recommender system.
    """

    def inner(self, *args, **kwargs):

        start = time.time_ns()

        f(self, *args, **kwargs)

        end = time.time_ns()

        self.build_time = end - start

    return inner


def rec_timer(f):
    """
    Decorator used to measure the time for a system to produce a recommendation and
    updates the object's store of all recommendation execution times, re-calculating the average.
    """

    def inner(self, *args, **kwargs):

        start = time.time_ns()

        ret = f(self, *args, **kwargs)

        end = time.time_ns()

        self.rec_times["times"].append(end - start)
        self.rec_times["avg"] = sum(self.rec_times["times"]) / len(
            self.rec_times["times"]
        )

        return ret

    return inner


class RecommenderSystem:
    """
    A base class for recommender systems that recommend using ingredients.
    """

    def __init__(self):
        # common attributes among all systems

        # unique ID for the class' instantiation
        self.execution_id = str(uuid.uuid4().hex)
        # filepaths of associated disk data
        self.disk_data = {}

        self.rec_times = {"times": [], "avg": 0}
        
    def build_ingredient_index(self, num_trees, out_path):
        
        ingredient_embeddings = []
        # get vectors for each ingredient using recipe vectorizer
        for ingredient in unique_ingredients:
            
            ingredient_embed = self.recipe_vectorizer([ingredient])
            
            ingredient_embeddings.append(ingredient_embed)

        self.build_index(iterable=ingredient_embeddings, num_trees=num_trees, out_path=out_path, recipe_index=False, save=True)
        
    def build_index(self,iterable, num_trees: int, out_path: str, recipe_index: bool=True, save: bool=True):
        
        # create index using class attributes
        index = AnnoyIndex(self.vec_size, self.index_distance_metric)
        
        # populate index
        for i, v in enumerate(iterable):
            index.add_item(i, v)
        
        # build and save
        index.build(num_trees)
        
        if save:
            index.save(out_path)
        
        # set to relevant class attribute
        if recipe_index:
            self.recipe_index = index
        else:
            self.ingredient_index = index

    def load_index(self, index_path: str):

        self.index = AnnoyIndex(self.vec_size, self.index_distance_metric)
        self.index.load(index_path)
                   

    @rec_timer
    def get_recommendations(
        self, recipe: List[str], n_recommendations: int = 10, search_id: int = None, get_recipes: bool=True
    ) -> pd.DataFrame:
        """
        Creates a recipe vector from a list of ingredients and queries the Annoy index for the `n_recommendations` nearest neighbours.

        Inputs:
            - `recipe`: `List[str]`, a list of string ingredients
            - `n_recommendations`: `int`, the number of recommendations to return
            - `get_recipes` : `bool`: if True, recommends recipes; if False, reccommends ingredients
        Outputs:
            - `pd.DataFrame`, a sorted DataFrame of the recommended recipes
        """

        try:

            # get the vector of the recipe
            recipe_vec = self.recipe_vectorizer(recipe)

            if get_recipes:
                # get closest vectors from the dataset
                rec_indexes = self.index.get_nns_by_vector(recipe_vec, n_recommendations)

            else:
                # get closest vectors from the ingredients dataset
                rec_indexes = self.ingredient_index.get_nns_by_vector(recipe_vec, n_recommendations)
                
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

            if get_recipes:
                # map recommendations to recipes
                recs = recipes.iloc[rec_indexes].copy()
            else:
                # map recommendations to ingredients
                recs = [unique_ingredients[i] for i in rec_indexes]    
            
            return recs

        except KeyError:
            raise ValueError(
                "One of the given ingredients did not exist in the training dataset."
            )

    def timer(self, func):

        start = time.time_ns()

        func()

        end = time.time_ns()

        self.build_time = end - start
