import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem, build_timer
from recipe_rec.utilities import check_file_exists, check_is_dir


class Recipe2Vec(RecommenderSystem):
    """
    Builds a recommender system using the Word2Vec language model, training the model and calculating recipe embeddings which
    are stored in an `AnnoyIndex`.

    Parameters:
        - `index_path: Path = None` (optional): the path to a pre-built AnnoyIndex.
        - `model_path: Path = None`: the path to a pre-trained model.
        - `num_epochs: int = 30`: the number of epochs to use in training.
        - `alpha: float = 0.025`: the learning rate to use in training.
        - `vec_size: int = 100`: the size of the model's embeddings.
        - `index_distance_metric: str = "manhattan"`: the distance metric to use when building the `AnnoyIndex`.
        - `output_dir: Path = pathlib.Path(".")`: the base directory to use when saving files.
        - `verbose: bool`: outputs updates during training if True, outputs nothing otherwise.

    """

    @build_timer
    def __init__(
        self,
        index_path: Path = None,
        model_path: Path = None,
        num_epochs: int = 30,
        alpha: float = 0.025,
        vec_size: int = 100,
        index_distance_metric="manhattan",
        output_dir: Path = Path("."),
        verbose: bool = True,
    ):

        super().__init__()

        self.num_epochs: int = num_epochs
        self.alpha: float = alpha
        self.vec_size: int = vec_size
        self.index_distance_metric: str = index_distance_metric
        self.output_dir: Path = check_is_dir(output_dir)
        self.verbose: bool = verbose

        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL)

        self.disk_data: Dict[str, Path] = {"index": index_path, "model": model_path}

        # check files exist
        for filepath in self.disk_data.values():

            if filepath is not None:

                check_file_exists(filepath)

        if self.disk_data["model"] is None:

            self.logger.info("Training Word2Vec model.")

            # need to build model and save vectors
            self.disk_data["model"] = self.train_model()

            self.logger.info(
                f"Trained Word2Vec model, stored at {self.disk_data['model']}"
            )
        else:

            self.logger.info("Loading pre-trained model.")
            # load model here
            self.model: Word2Vec = Word2Vec.load(self.disk_data["model"])

            self.logger.info("Loaded pre-trained model.")

        if self.disk_data["index"] is None:

            self.logger.info("Building an index for the recipes using trained model.")

            out_path: Path = Path(
                self.output_dir, f"recipe2vec_{self.execution_id}.ann"
            )

            recipe_vectors: pd.Series = recipes["RecipeIngredientParts"].apply(
                self.recipe_vectorizer
            )

            # need to build an index
            self.build_index(
                iterable=recipe_vectors,
                num_trees=10,
                out_path=out_path,
                recipe_index=True,
            )
            self.disk_data["index"] = out_path

            self.logger.info(
                f"Built Annoy index and saved to {self.disk_data['index']}"
            )

        else:

            self.logger.info("Loading pre-built Annoy index.")
            self.load_index(self.disk_data["index"])

    def train_model(self) -> Path:
        """
        Trains the Word2Vec model on the recipe corpus.

        Returns:
            - `pathlib.Path`: the path to the model's binary file written on disk.
        """

        self.training_losses: List[float] = []

        # temporarily change root logger level to hide excessive gensim outputs
        prev_log_level: int = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.WARNING)

        try:
            # create model
            self.model: Word2Vec = Word2Vec(
                recipes["RecipeIngredientParts"].values,
                # use skipgram, not CBOW
                sg=1,
                vector_size=self.vec_size,
                alpha=self.alpha,
                # ensures rarely-occurring ingredients still are given a vector
                min_count=1,
                epochs=self.num_epochs,
                compute_loss=True,
                callbacks=[get_loss_callback(self)],
            )

            # write to file
            model_path: Path = Path(
                self.output_dir, f"recipe2vec_{self.execution_id}.model"
            )
            self.model.save(model_path)

            return model_path

        except Exception as e:
            self.logger.warn(e)

        # restore root logger back to normal level
        logging.getLogger().setLevel(prev_log_level)

    def recipe_vectorizer(self, recipe: List[str]) -> np.array:
        """
        Maps a list of ingredients in a recipe to the average of each ingredient's embedding vector.

        Parameters:
            - `recipe: List[str]`: a list of ingredient tokens which comprise the recipe.

        Returns:
            - `np.array`: the recipe's embedding.
        """

        ingredient_vecs: np.array = np.array(
            [self.model.wv[ingredient] for ingredient in recipe]
        )

        recipe_vec: np.array = np.mean(ingredient_vecs, axis=0)

        return recipe_vec


class get_loss_callback(CallbackAny2Vec):
    """
    Gets training loss after each epoch and logs the current epoch number.

    Parameters:
        - `rec_system: Recipe2Vec`: instance of Recipe2Vec to store training losses in.
    """

    def __init__(self, rec_system: Recipe2Vec):
        self.epoch: int = 1
        self.rec_system: Recipe2Vec = rec_system

    def on_epoch_end(self, model: Word2Vec):
        """
        Gets training losses at the end of an epoch.

        Parameters:
            - `model: Word2Vec`: the Word2Vec model being trained.
        """

        self.rec_system.training_losses.append(model.get_latest_training_loss())

        self.rec_system.logger.info(f"Completed epoch {self.epoch}.")

        self.epoch += 1
