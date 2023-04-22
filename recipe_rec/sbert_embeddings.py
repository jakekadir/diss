import logging
import pathlib
import pickle
import uuid
from typing import List

import numpy as np
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem, build_timer


class SBERTRecommender(RecommenderSystem):
    @build_timer
    def __init__(
        self,
        embeddings_path: str = None,
        index_path: str = None,
        verbose: bool = True,
        index_distance_metric: str = "manhattan",
    ):

        super().__init__()

        self.index_distance_metric = index_distance_metric
        self.verbose = verbose
        # constants
        self.embedding_col = "RecipeIngredientParts"
        self.vec_size = 384

        self.disk_data = {"embeddings": embeddings_path, "index": index_path}

        # load the transformer model
        transformer_model: str = "all-MiniLM-L12-v2"
        self.model: SentenceTransformer = SentenceTransformer(transformer_model)

        if self.verbose:
            logging.basicConfig(
                format="%(levelname)s - %(asctime)s: %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )

        if embeddings_path is None:

            if verbose:
                logging.info("Generating embeddings for the recipe dataset.")
            # generate embeddings for ingredients
            embeddings_path: str = self.generate_embeddings()

            if verbose:
                logging.info(f"Generated recipe embeddings at {embeddings_path}")
        else:

            if verbose:
                logging.info("Loading recipe embeddings from disk.")
            # load embeddings?
            with open(embeddings_path, "rb") as f:
                self.ingredient_embeddings = pickle.load(f)

        if index_path is None:

            if verbose:
                logging.info("Building Annoy index from embeddings.")

            out_path = f"./recipe_rec/data/sbert_{self.execution_id}.ann"

            # create index
            built_index_path: str = self.build_index(
                iterable=self.ingredient_embeddings,
                num_trees=10,
                out_path=out_path,
                recipe_index=True,
                save=True
            )

            if verbose:
                logging.info(f"Built Annoy Index at {built_index_path}")
        else:

            if verbose:
                logging.info("Loading Annoy index from disk.")
            # load index
            self.load_index(index_path)

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients to a vector.

        """

        # combine ingredients into a string
        joined_ingredients = ",".join(recipe)

        # retrieve BERT vector for string
        recipe_vec = self.model.encode(
            joined_ingredients, show_progress_bar=self.verbose
        )

        return recipe_vec

    def generate_embeddings(self) -> str:

        # generate embeddings
        self.ingredient_embeddings = self.model.encode(
            recipes[self.embedding_col].values, show_progress_bar=self.verbose
        )

        embeddings_path: str = f"./data/sbert_recipe_embeddings{self.execution_id}.pkl"
        with open(embeddings_path, "wb") as f:

            pickle.dump(self.ingredient_embeddings, f)

        return embeddings_path

    # def build_index(self) -> str:

    #     """
    #     Takes a path to a dataset, loads the data and produces sentence-BERT embeddings
    #     for a given column of the dataset.

    #     An Annoy Index is constructed for these embeddings and written to a file
    #     which incorporates the execution_id in the filename.

    #     """

    #     self.index = AnnoyIndex(self.vec_size, self.index_distance_metric)

    #     for i in enumerate(self.ingredient_embeddings):
    #         self.index.add_item(i, self.ingredient_embeddings[i])

        out_path = f"./recipe_rec/data/sbert_{self.execution_id}.ann"

    #     self.index.build(10)
    #     self.index.save(out_path)

    #     return out_path

    def load_index(self, index_path: str):

        self.index = AnnoyIndex(self.vec_size, self.index_distance_metric)
        self.index.load(index_path)
