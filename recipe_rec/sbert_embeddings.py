import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from recipe_rec.recommender_system import RecommenderSystem, build_timer
from recipe_rec.utilities import check_file_exists, check_is_dir


class SBERTRecommender(RecommenderSystem):
    """
    A recommender system using pre-trained SBERT embeddings. Generates embeddings for a string of each recipe's ingredients before placing
    the embeddings into an `AnnoyIndex`.

    Parameters:
        - `embeddings_path: pathlib.Path = None` (optional): the path of pre-generated SBERT embeddings.
        - `index_path: Path = None` (optional): the path of a pre-built `AnnoyIndex`.
        - `index_distance_metric: str = "manhattan"` (optional): the distance metric to use when building the `AnnoyIndex`.
        - `output_dir: pathlib.Path = pathlib.Path(".")` (optional): the base directory to use when writing files to disk.
        - `verbose: bool = True` (optional): outputs updates during training if True, outputs nothing otherwise.
    """

    @build_timer
    def __init__(
        self,
        embeddings_path: Path = None,
        index_path: Path = None,
        index_distance_metric: str = "manhattan",
        output_dir: Path = Path("."),
        verbose: bool = True,
    ) -> None:

        super().__init__()

        self.index_distance_metric: str = index_distance_metric
        # constants
        self.embedding_col: str = "RecipeIngredientParts"
        self.vec_size: int = 384
        self.output_dir: Path = check_is_dir(output_dir)
        self.verbose: bool = verbose

        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL)

        self.disk_data: Dict[str, Path] = {
            "embeddings": embeddings_path,
            "index": index_path,
        }
        # validate provided file paths before trying to do anything
        for filepath in self.disk_data.values():
            if filepath is not None:
                check_file_exists(filepath)

        # load the transformer model
        transformer_model: str = "all-MiniLM-L12-v2"
        self.model: SentenceTransformer = SentenceTransformer(transformer_model)

        if embeddings_path is None:

            self.logger.info("Generating embeddings for the recipe dataset.")
            # generate embeddings for ingredients
            embeddings_path: Path = self.generate_embeddings()
            self.disk_data["embeddings"] = embeddings_path
            self.logger.info(f"Generated recipe embeddings at {embeddings_path}")
        else:

            self.logger.info("Loading recipe embeddings from disk.")
            # load embeddings
            with open(self.disk_data["embeddings"], "rb") as f:
                self.ingredient_embeddings: np.ndarray = pickle.load(f)

        if index_path is None:

            self.logger.info("Building Annoy index from embeddings.")

            out_path: Path = Path(self.output_dir, f"sbert_{self.execution_id}.ann")

            # create index
            built_index_path: str = self.build_index(
                iterable=self.ingredient_embeddings,
                num_trees=10,
                out_path=out_path,
                recipe_index=True,
                save=True,
            )

            self.logger.info(f"Built Annoy Index at {built_index_path}")
        else:

            self.logger.info("Loading Annoy index from disk.")
            # load index
            self.load_index(index_path)

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients in a recipe to the SBERT embedding of the concatenated string of these ingredients.

        Parameters:
        `recipe: List[str]`: a list of ingredient tokens which comprise the recipe.

        Returns:
        `np.array`: the recipe's embedding.
        """

        # combine ingredients into a string
        joined_ingredients = ",".join(recipe)

        # retrieve BERT vector for string
        recipe_vec = self.model.encode(
            joined_ingredients, show_progress_bar=self.verbose
        )

        return recipe_vec

    def generate_embeddings(self) -> Path:
        """
        Generates SBERT embeddings for the entire recipe dataset, stored at the `ingredient_embeddings` attribute of the class.

        Returns:
            - `pathlib.Path`: the path to which the generated embeddings are written.

        """
        concatenated_ingredients: pd.Series = self.recipes[self.embedding_col].str.join(
            " "
        )

        # generate embeddings
        self.ingredient_embeddings: np.ndarray = self.model.encode(
            concatenated_ingredients, show_progress_bar=self.verbose
        )

        embeddings_path: Path = Path(
            self.output_dir, f"sbert_recipe_embeddings{self.execution_id}.pkl"
        )

        with open(embeddings_path, "wb") as f:

            pickle.dump(self.ingredient_embeddings, f)

        return embeddings_path
