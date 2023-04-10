import logging
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec

from recipe_rec import recipes
from recipe_rec.recommender_system import IngredientRecommender


class Recipe2Vec(IngredientRecommender):
    def __init__(
        self,
        index_path: str = None,
        model_path: str = None,
        vec_size: int = 100,
        index_distance_metric="manhattan",
        verbose: bool = True,
    ):

        super().__init__()
        
        self.verbose = verbose
        self.vec_size = vec_size
        self.index_distance_metric = index_distance_metric

        self.disk_data = {"index": index_path, "model": model_path}

        if self.verbose:
            logging.basicConfig(
                format="%(levelname)s - %(asctime)s: %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )

        if self.disk_data["model"] is None:

            if verbose:
                logging.info("Training Word2Vec model.")

            # need to build model and save vectors
            self.disk_data["model"]: str = self.train_model()

            if verbose:
                logging.info(
                    f"Trained Word2Vec model, stored at {self.disk_data['model']}"
                )
        else:

            if verbose:
                logging.info("Loading pre-trained model.")
            # load model here
            self.model = Word2Vec.load(self.disk_data["model"])

            if verbose:
                logging.info("Loaded pre-trained model.")

        if self.disk_data["index"] is None:

            if verbose:
                logging.info("Building an index for the recipes using trained model.")
            # need to build an index
            self.disk_data["index"]: str = self.build_index()

            if verbose:
                logging.info(
                    f"Built Annoy index and saved to {self.disk_data['index']}"
                )

        else:

            if verbose:
                logging.info("Loading pre-built Annoy index.")
            self.load_index(self.disk_data["index"])

    def train_model(self) -> str:

        # create model
        self.model = Word2Vec(
            recipes["RecipeIngredientParts"].values,
            # use skipgram, not CBOW
            sg=1,
            vector_size=self.vec_size,
            # ensures rarely-occurring ingredients still are given a vector
            min_count=1,
            epochs=30,
        )

        model_path = f"./data/recipe2vec_{self.execution_id}.model"
        self.model.save(model_path)

        return model_path

    def build_index(self) -> AnnoyIndex:

        """
        Takes a path to a dataset, loads the data and produces recipe2vec embeddings
        for a given column of the dataset.

        An Annoy Index is constructed for these embeddings and written to a file
        which incorporates the execution_id in the filename.

        """

        if self.verbose:
            logging.info("Generating vectors for recipes.")
        # map the recipes to vectors
        recipe_vectors: pd.Series = recipes["RecipeIngredientParts"].apply(
            self.recipe_vectorizer
        )

        # build an index
        self.index = AnnoyIndex(self.vec_size, self.index_distance_metric)

        if self.verbose:
            logging.info("Populating Annoy index.")
        # populate
        for vec_index, vec in recipe_vectors.items():
            self.index.add_item(vec_index, vec)

        if self.verbose:
            logging.info("Storing index on disk.")

        # build and save
        out_path = f"./data/recipe2vec_{self.execution_id}.ann"

        self.index.build(10)
        self.index.save(out_path)

        return out_path

    def load_index(self, index_path: str):

        self.index = AnnoyIndex(self.vec_size, self.index_distance_metric)
        self.index.load(index_path)

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients in a recipe to the average of each ingredient's embedding vector.
        """

        ingredient_vecs = np.array([self.model.wv[ingredient] for ingredient in recipe])

        recipe_vec = np.mean(ingredient_vecs, axis=0)

        return recipe_vec
