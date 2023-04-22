import logging
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tqdm import tqdm

from recipe_rec import recipes
from recipe_rec.recommender_system import RecommenderSystem, build_timer

RANDOM_STATE = 42


class FeatureGenerationRecommender(RecommenderSystem):
    @build_timer
    def __init__(
        self,
        embeddings_path: str = None,
        classifiers_path: str = None,
        labelled_dataset_path: str = None,
        index_distance_metric: str = "manhattan",
        verbose: bool = True,
        index_path: str = None,
    ):

        super().__init__()

        self.labelled_cols = [
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
        self.vec_size = 19
        self.verbose = verbose

        self.index_distance_metric = index_distance_metric
        self.embedding_col = "RecipeIngredientParts"

        # load the transformer model
        transformer_model: str = "all-MiniLM-L12-v2"
        self.bert_encoder: SentenceTransformer = SentenceTransformer(transformer_model)

        if self.verbose:
            logging.basicConfig(
                format="%(levelname)s - %(asctime)s: %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )

        # map inputs to object attribute
        self.disk_data = {
            "embeddings": embeddings_path,
            "classifiers": classifiers_path,
            "labelled_data": labelled_dataset_path,
            "index": index_path,
        }

        # if no ingredient embeddings are given, make them
        if self.disk_data["embeddings"] is None:

            if verbose:
                logging.info("Generating embeddings for the recipe dataset.")
            # generate embeddings for ingredients
            self.disk_data["embeddings"] = self.generate_embeddings()

            if verbose:
                logging.info(
                    f"Generated recipe embeddings at {self.disk_data['embeddings']}"
                )
        else:

            if verbose:
                logging.info("Loading recipe embeddings from disk.")
            # load embeddings?
            with open(self.disk_data["embeddings"], "rb") as f:
                self.ingredient_embeddings = pickle.load(f)

        # if no trained classifiers are given
        if self.disk_data["classifiers"] is None:

            # train classifiers
            if verbose:
                logging.info("Training classifiers.")
            self.disk_data["classifiers"]: str = self.train_classifiers()

            if verbose:
                logging.info(f"Saved classifiers to {self.disk_data['classifiers']}")
        else:

            if verbose:
                logging.info("Loading pre-trained classifiers.")
            # load classifiers
            with open(classifiers_path, "rb") as f:
                self.classifiers = pickle.load(f)

        if labelled_dataset_path is None:

            if verbose:
                logging.info("Labelling dataset with classifiers.")
            # label the dataset
            self.disk_data["labelled_data"]: str = self.label_dataset()

            if verbose:
                logging.info(
                    f"Labelled dataset and saved to disk at {self.disk_data['labelled_data']}"
                )
        else:

            if verbose:
                logging.info("Loading pre-labelled dataset.")
            # load dataset
            self.load_labelled_dataset(labelled_dataset_path)

        if index_path is None:
            np_dataset = self.labelled_dataset.to_numpy()
            out_path = f"./recipe_rec/data/feature_generation_{self.execution_id}.ann"

            # build an index
            self.disk_data["index"]: str = self.build_index(
                iterable=np_dataset,
                num_trees=10,
                out_path=out_path,
                recipe_index=True,
                save=True
                
            )
            if verbose:
                logging.info(f"Built index at {self.disk_data['index']}")
        else:

            # load the index
            if verbose:
                logging.info("Loading index from disk.")
            self.load_index(index_path)

    def generate_embeddings(self) -> str:

        # generate embeddings
        self.ingredient_embeddings = self.bert_encoder.encode(
            recipes[self.embedding_col].values
        )

        embeddings_path: str = f"./data/sbert_recipe_embeddings{self.execution_id}.pkl"
        with open(embeddings_path, "wb") as f:

            pickle.dump(self.ingredient_embeddings, f)

        return embeddings_path

    def train_classifiers(self) -> str:

        # import labelled dataset
        labelled_df = pd.read_csv(
            "./feature_generation/trimmed_labelled_slim.csv", nrows=300
        )

        # remove double quotes from ingredient string
        labelled_df["RecipeIngredientParts"] = labelled_df[
            "RecipeIngredientParts"
        ].str.replace('"', "")

        # encode ingredient string
        encoded_ingredients = self.bert_encoder.encode(
            labelled_df["RecipeIngredientParts"].values
        )

        # bundle into a training data dict
        training_data = {"ingredient_vector": encoded_ingredients}

        for col in self.labelled_cols:
            training_data[col] = labelled_df[col].values

        metrics = {
            "col_name": [],
            "accuracy": [],
            "f1": [],
            "precision": [],
            "recall": [],
        }

        if self.verbose:
            logging.info("About to begin training.")

        self.classifiers = {}

        # prep and split dataset
        X = training_data["ingredient_vector"]
        Y = training_data[col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.30, random_state=RANDOM_STATE
        )

        for col in self.labelled_cols:

            # initialise and fit model
            classifier_model = RandomForestClassifier()

            params = {
                "n_estimators": [50, 100, 150, 300, 500],
                "criterion": ["gini", "entropy"],
            }

            random_search = RandomizedSearchCV(
                classifier_model, params, random_state=RANDOM_STATE
            )

            best_model = random_search.fit(X_train, y_train)

            # assess model predictions
            y_pred = best_model.predict(X_test)

            metrics["col_name"].append(col)
            self.classifiers[col] = best_model
            metrics["f1"].append(f1_score(y_test, y_pred))
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred))
            metrics["recall"].append(recall_score(y_test, y_pred))

        self.training_metrics = pd.DataFrame(metrics)

        if self.verbose:
            logging.info("Clasifiers trained.")

        classifier_out_path: str = (
            f"./recipe_rec/data/trained_classifiers_{self.execution_id}.pkl"
        )
        with open(classifier_out_path, "wb") as f:

            pickle.dump(self.classifiers, f)

        return classifier_out_path

    def label_dataset(self):

        if self.verbose:
            logging.info("Predicting labels for food attributes.")

        for col in tqdm(self.labelled_cols):

            recipes[col] = self.classifiers[col].predict(self.ingredient_embeddings)

        self.labelled_dataset = recipes[self.labelled_cols]

        labelled_path_out: str = (
            f"./recipe_rec/data/labelled_dataset_{self.execution_id}.csv"
        )
        self.labelled_dataset.to_csv(labelled_path_out)

        return labelled_path_out

    def load_labelled_dataset(self, dataset_path: str):

        self.labelled_dataset = pd.read_csv(dataset_path, index_col=0)

    def recipe_vectorizer(self, ingredients: List[str]):

        ingredient_str: str = ",".join(ingredients)

        ingredient_embed = self.bert_encoder.encode(
            ingredient_str, show_progress_bar=self.verbose
        )

        recipe_vec = []

        for col in self.labelled_cols:

            recipe_vec.append(self.classifiers[col].predict([ingredient_embed])[0])

        return np.array(recipe_vec)
