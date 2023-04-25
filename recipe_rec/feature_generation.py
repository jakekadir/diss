import logging
import pickle
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm

from recipe_rec.data import recipes
from recipe_rec.recommender_system import RecommenderSystem, build_timer
from recipe_rec.utilities import check_file_exists, check_is_dir

RANDOM_STATE = 42


class FeatureGenerationRecommender(RecommenderSystem):
    """
    Builds a recommender system using a manually-labelled dataset of new features, training
    classifiers to predict these features using SBERT embeddings of the recipes' ingredients.

    These new features are used to create a recipe embedding space from which recommendations can be made.

    Parameters:
    - `embeddings_path: pathlib.Path = None` (optional): a path to a binary file of pre-calculated SBERT embeddings for the entire recipe dataset.
    - `classifiers_path: pathlib.Path = None` (optional): a path to a binary file of pre-trained classifiers.
    - `prelabelled_dataset_path: pathlib.Path = None`: a path to the pre-labelled dataset.
    - `index_path: pathlib.Path = None` (optional): a path to a pre-built AnnoyIndex.
    - `index_distance_metric: str = "manhattan"` (optional): the distance metric to use when building an AnnoyIndex
    - `output_dir: pathlib.Path = pathlib.Path(".")` (optional): the base path to write all files to.
    - `verbose: bool = False`: outputs updates during training if True, outputs nothing otherwise.
    """

    @build_timer
    def __init__(
        self,
        embeddings_path: Path = None,
        classifiers_path: Path = None,
        prelabelled_dataset_path: Path = None,
        labelled_dataset_path: Path = None,
        index_path: Path = None,
        index_distance_metric: str = "manhattan",
        output_dir: Path = Path("."),
        verbose: bool = False,
    ) -> None:

        super().__init__()

        self.labelled_cols: List[str] = [
            "Savoury",
            "Rough",
            "Hot",
            "Spicy",
            "Acidic",
            "Crunchy",
            "Creamy",
            "Sticky",
            "Liquid",
            "Aromatic",
            "Salty",
            "Citrusy",
            "Herbal",
            "Fluffy",
            "Flaky",
            "Cooling",
            "Chunky",
            "Fishy",
            "Firm",
        ]
        self.vec_size: int = 19
        self.output_dir: Path = check_is_dir(output_dir)
        self.verbose: bool = verbose

        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.CRITICAL)

        self.index_distance_metric: str = index_distance_metric
        self.embedding_col: str = "RecipeIngredientParts"

        # load the transformer model
        transformer_model: str = "all-MiniLM-L12-v2"
        self.bert_encoder: SentenceTransformer = SentenceTransformer(transformer_model)

        # map inputs to object attribute
        self.disk_data: Dict[str, Path] = {
            "embeddings": embeddings_path,
            "classifiers": classifiers_path,
            "prelabelled_data": check_file_exists(prelabelled_dataset_path),
            "labelled_data": labelled_dataset_path,
            "index": index_path,
        }

        # check any provided filepaths exist
        for filepath in self.disk_data.values:

            if filepath is not None:

                check_file_exists(filepath)

        # if no ingredient embeddings are given, make them
        if self.disk_data["embeddings"] is None:

            self.logger.info("Generating embeddings for the recipe dataset.")
            # generate embeddings for ingredients
            self.disk_data["embeddings"] = self.generate_embeddings()

            self.logger.info(
                f"Generated recipe embeddings at {self.disk_data['embeddings']}"
            )
        else:

            self.logger.info("Loading recipe embeddings from disk.")
            # load embeddings?
            with open(self.disk_data["embeddings"], "rb") as f:

                self.ingredient_embeddings: np.ndarray = pickle.load(f)

        # if no trained classifiers are given
        if self.disk_data["classifiers"] is None:

            # train classifiers
            self.logger.logging.info("Training classifiers.")
            self.disk_data["classifiers"] = self.train_classifiers()

            self.logger.info(f"Saved classifiers to {self.disk_data['classifiers']}")
        else:

            self.logger.info("Loading pre-trained classifiers.")
            # load classifiers
            with open(classifiers_path, "rb") as f:
                self.classifiers: Dict[str, RandomForestClassifier] = pickle.load(f)

        if labelled_dataset_path is None:

            self.logger.info("Labelling dataset with classifiers.")
            # label the dataset
            self.disk_data["labelled_data"] = self.label_dataset()

            self.logger.info(
                f"Labelled dataset and saved to disk at {self.disk_data['labelled_data']}"
            )
        else:

            self.logger.info("Loading pre-labelled dataset.")
            # load dataset
            self.labelled_dataset: pd.DataFrame = pd.read_csv(
                labelled_dataset_path, index_col=0
            )

        if index_path is None:
            np_dataset: np.ndarray = self.labelled_dataset.to_numpy()
            out_path: Path = Path(
                self.output_dir, f"feature_generation_{self.execution_id}.ann"
            )

            # build an index
            self.disk_data["index"]: str = self.build_index(
                iterable=np_dataset,
                num_trees=10,
                out_path=out_path,
                recipe_index=True,
                save=True,
            )
            self.logger.info(f"Built index at {self.disk_data['index']}")
        else:

            # load the index
            self.logger.info("Loading index from disk.")

            self.load_index(index_path)

    def generate_embeddings(self) -> Path:
        """
        Generates SBERT embeddings for the string list of each recipe's ingredients, writing to a pickle file.

        Returns:
        `pathlib.Path`: the path to the pickle file containing the recipe embeddings.
        """

        # combine ingredients into a single string separted by spaces
        concatenated_ingredients: pd.Series = recipes[self.embedding_col].str.join(" ")

        # generate embeddings
        self.ingredient_embeddings: np.ndarray = self.bert_encoder.encode(
            concatenated_ingredients
        )

        embeddings_path: Path = Path(
            self.output_dir, f"sbert_recipe_embeddings{self.execution_id}.pkl"
        )
        with open(embeddings_path, "wb") as f:

            pickle.dump(self.ingredient_embeddings, f)

        return embeddings_path

    def train_classifiers(self) -> Path:
        """
        Trains classifiers for each manually-labelled feature and saves the fitted models to a binary file.

        Returns:
        `pathlib.Path`: the path to the saved fitted models.
        """

        # import labelled dataset
        labelled_df: pd.DataFrame = pd.read_csv(
            self.disk_data["prelabelled_data"], nrows=300
        )

        # remove double quotes from ingredient string
        labelled_df["RecipeIngredientParts"] = labelled_df[
            "RecipeIngredientParts"
        ].str.replace('"', "")

        # encode ingredient string
        encoded_ingredients: np.ndarray = self.bert_encoder.encode(
            labelled_df["RecipeIngredientParts"].values
        )

        # bundle into a training data dict
        training_data: Dict[str, np.ndarray] = {
            "ingredient_vector": encoded_ingredients
        }

        for col in self.labelled_cols:
            training_data[col] = labelled_df[col].values

        metrics: Dict[str, Union[List[float], List[str]]] = {
            "col_name": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        self.logger.info("About to begin training.")

        self.classifiers: Dict[str, RandomForestClassifier] = {}

        # prep and split dataset
        X: np.ndarray = training_data["ingredient_vector"]
        Y: np.ndarray = training_data[col]

        X_train: np.ndarray
        X_test: np.ndarray
        y_train: np.ndarray
        y_test: np.ndarray
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=RANDOM_STATE
        )

        for col in self.labelled_cols:

            # initialise and fit model
            classifier_model: RandomForestClassifier = RandomForestClassifier()

            params: Dict[str, Union[List[int], List[str]]] = {
                "n_estimators": [50, 100, 150, 300, 500],
                "criterion": ["gini", "entropy"],
            }

            random_search: RandomizedSearchCV = RandomizedSearchCV(
                classifier_model, params, random_state=RANDOM_STATE
            )

            best_model: RandomForestClassifier = random_search.fit(X_train, y_train)

            # assess model predictions
            y_pred: np.ndarray = best_model.predict(X_test)

            metrics["col_name"].append(col)
            self.classifiers[col] = best_model
            metrics["f1"].append(f1_score(y_test, y_pred))
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred))
            metrics["recall"].append(recall_score(y_test, y_pred))

        self.training_metrics: pd.DataFrame = pd.DataFrame(metrics)

        self.logger.info("Clasifiers trained.")

        classifier_out_path: Path = Path(
            self.ouput_dir, f"trained_classifiers_{self.execution_id}.pkl"
        )
        with open(classifier_out_path, "wb") as f:

            pickle.dump(self.classifiers, f)

        return classifier_out_path

    def label_dataset(self) -> Path:
        """
        Predicts labels for the entire dataset using the fitted classifiers.

        Returns:
        `pathlib.Path`: the path to the fully-labelled dataset.
        """

        self.logger.info("Predicting labels for food attributes.")

        for col in tqdm(self.labelled_cols):

            recipes[col] = self.classifiers[col].predict(self.ingredient_embeddings)

        self.labelled_dataset: pd.DataFrame = recipes[self.labelled_cols]

        labelled_path_out: Path = Path(
            self.output_dir, f"labelled_dataset_{self.execution_id}.csv"
        )
        self.labelled_dataset.to_csv(labelled_path_out)

        return labelled_path_out

    def recipe_vectorizer(self, ingredients: List[str]) -> np.array:
        """
        Generates an SBERT embedding for the given ingredients and predicts features using pre-trained classifiers.

        Parameters:
        `ingredients: List[str]`: a list of ingredients to use in producing the embedding.
        Returns:
        `np.array`: the embedding for the given recipe.
        """

        ingredient_str: str = ",".join(ingredients)

        ingredient_embed: np.ndarray = self.bert_encoder.encode(
            ingredient_str, show_progress_bar=self.verbose
        )

        recipe_vec: List[int] = []

        for col in self.labelled_cols:

            recipe_vec.append(self.classifiers[col].predict([ingredient_embed])[0])

        return np.array(recipe_vec)
