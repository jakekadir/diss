import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from gensim.models import FastText

from recipe_rec.recommender_system import RecommenderSystem, build_timer
from recipe_rec.utilities import check_file_exists, check_is_dir, space_replacer


class fastRecipeRecommender(RecommenderSystem):
    """
    Builds a recommender system using the fastText language model, training the model and calculating recipe embeddings which
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
        training_data_txt: Path = None,
        model_path: Path = None,
        num_epochs: int = 30,
        alpha: float = 0.025,
        vec_size: int = 100,
        index_distance_metric="manhattan",
        output_dir: Path = Path("."),
        verbose: bool = False,
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

        self.disk_data: Dict[str, Path] = {
            "index": index_path,
            "model": model_path,
            "training_data_txt": training_data_txt,
        }

        # check files exist
        for filepath in self.disk_data.values():

            if filepath is not None:

                check_file_exists(filepath)

        # if self.disk_data["training_data_txt"] is None:

        #     # generate train data
        #     self.disk_data["training_data_txt"] = self.generate_train_data()

        # if there is training data, do nothing as only the path is needed for fastText

        if self.disk_data["model"] is None:

            self.logger.info("Training fastText model.")

            # need to build model and save vectors
            self.disk_data["model"] = self.train_model()

            self.logger.info(
                f"Trained fastText model, stored at {self.disk_data['model']}"
            )
        else:

            self.logger.info("Loading pre-trained model.")
            # load model here
            self.model = FastText.load(self.disk_data["model"])

            self.logger.info("Loaded pre-trained model.")

        if self.disk_data["index"] is None:

            self.logger.info("Building an index for the recipes using trained model.")

            out_path: Path = Path(self.output_dir, f"fastRecipe{self.execution_id}.ann")

            recipe_vectors: pd.Series = self.recipes["RecipeIngredientParts"].apply(
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

    def generate_train_data(self) -> Path:

        ingredients: pd.Series = self.recipes["RecipeIngredientParts"].apply(
            space_replacer
        )
        ingredients = ingredients.apply(lambda tup: " ".join(tup))

        train_data_path: Path = Path(
            self.output_dir, f"fastText_train_data_{self.execution_id}.txt"
        )

        np.savetxt(train_data_path, ingredients.values, fmt="%s", delimiter=" ")

        return train_data_path

    def train_model(self) -> Path:
        """
        Trains the fastText model on the recipe corpus.

        Returns:
            - `pathlib.Path`: the path to the model's binary file written on disk.
        """

        self.training_losses: List[float] = []

        # temporarily change root logger level to hide excessive gensim outputs
        prev_log_level: int = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.WARNING)

        window_size = 10
        num_epochs = 300
        min_count = 5
        num_neg_samples = 5
        try:

            # convert path to str; fastText can't take Path
            # training_data_str: str = self.disk_data["training_data_txt"].absolute().as_posix()

            ingredients: pd.Series = self.recipes["RecipeIngredientParts"].apply(
                space_replacer
            )

            # create model
            self.model: FastText = FastText(sentences=ingredients)

            # self.model: fasttext._fastText  = fasttext.train_unsupervised(
            #     input=training_data_str,
            #     model="skipgram",
            #     dim=self.vec_size,
            #     ws=window_size,
            #     epoch=num_epochs,
            #     minCount=min_count,
            #     neg=num_neg_samples,
            #     loss="ns",  # negative sampling
            #     verbose=self.verbose,
            # )

            # write to file
            model_path: Path = (
                Path(self.output_dir, f"recipe2vec_{self.execution_id}.model")
                .absolute()
                .as_posix()
            )

            self.model.save(model_path)

            return model_path

        except Exception as e:
            raise e
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

        recipe = space_replacer(recipe)

        ingredient_vecs: np.array = np.array(
            [self.model.wv[ingredient] for ingredient in recipe]
        )

        recipe_vec: np.array = np.mean(ingredient_vecs, axis=0)

        return recipe_vec
