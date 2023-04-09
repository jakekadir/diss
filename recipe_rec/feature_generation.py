import pandas as pd
from recommender_system import IngredientRecommender
from data_loader import get_recipes
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np

tqdm.pandas()

RANDOM_STATE = 42


class FeatureGenerationRecommender(IngredientRecommender):
    def __init__(self, dataset_path: Path, col_to_embed: str, execution_id: str):

        super().__init__("FeatureGeneration", "", 19)

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

    def train_classifiers(self):

        # import labelled dataset
        labelled_df = pd.read_csv(
            "./feature_generation/trimmed_labelled_slim.csv", nrows=300
        )

        # remove double quotes from ingredient string
        labelled_df["RecipeIngredientParts"] = labelled_df[
            "RecipeIngredientParts"
        ].str.replace('"', "")

        model_name = "paraphrase-MiniLM-L6-v2"

        # encode ingredient string
        self.bert_encoder = SentenceTransformer(model_name)
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

        print("About to begin training.")

        self.classifiers = {}

        for col in tqdm(self.labelled_cols):

            # prep and split dataset
            X = training_data["ingredient_vector"]
            Y = training_data[col]

            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.30, random_state=RANDOM_STATE
            )

            if np.isnan(y_train).any() or np.isnan(y_test).any():
                print(col, "contains NaN")
            # initialise and fit model
            classifier_model = RandomForestClassifier()
            classifier_model.fit(X_train, y_train)

            # assess model predictions
            y_pred = classifier_model.predict(X_test)

            metrics["col_name"].append(col)
            self.classifiers[col] = classifier_model
            metrics["f1"].append(f1_score(y_test, y_pred))
            metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            metrics["precision"].append(precision_score(y_test, y_pred))
            metrics["recall"].append(recall_score(y_test, y_pred))

        metrics_df = pd.DataFrame(metrics)

        print(metrics_df)

    def generate_features(self, verbose=True):
        # load the full dataset
        full_dataset = get_recipes("./data/recipes.csv")

        if verbose:
            print(f"Loaded {full_dataset.shape} recipes...")

        # turn ingredients back into a string
        full_dataset["RecipeIngredientParts"] = full_dataset[
            "RecipeIngredientParts"
        ].str.join(",")

        # feature_predictor = lambda row, column: self.classifiers[column].predict(self.bert_encoder.encode(row["IngredientSBERTVector"]))
        if verbose:
            print("Encoding ingredients to SBERT vectors:")
        encoded_ingredients = self.bert_encoder.encode(
            full_dataset["RecipeIngredientParts"].values
        )
        if verbose:
            print("Predicting labels for food attributes")
        for col in tqdm(self.labelled_cols):

            full_dataset[col] = self.classifiers[col].predict(encoded_ingredients)

        if verbose:
            print("Writing to CSV")
        full_dataset.to_csv("predicted_features.csv")


recommender = FeatureGenerationRecommender("", "", "")
recommender.train_classifiers()
recommender.generate_features()
