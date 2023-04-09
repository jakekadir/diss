from pathlib import Path
from annoy import AnnoyIndex
from data_loader import get_recipes
from typing import List
from recommender_system import IngredientRecommender
import pandas as pd
from glove import Glove, Corpus
import numpy as np


class RecipeGlove(IngredientRecommender):
    def __init__(self, dataset_path: Path, verbose: bool, execution_id: str):
        # set parent attributes
        super().__init__("RecipeGlove", "", 100)

    def generate_index(
        self, dataset_path: str, verbose: bool, execution_id: str
    ) -> AnnoyIndex:

        recipe_df = get_recipes(dataset_path)

        self.recipes = recipe_df["RecipeIngredientParts"].apply(
            lambda tuple: list(tuple)
        )

        # hyperparameters
        vec_size = 100
        learning_rate = 0.001
        num_epochs = 25

        corpus = Corpus()
        corpus.fit(self.recipes["RecipeIngred"])
        corpus.save(f"glove_corpus_{execution_id}.model")

        self.model = Glove(no_components=vec_size, learning_rate=learning_rate)

        self.model.fit(corpus.matrix, epochs=num_epochs, verbose=verbose)
        self.model.add_dictionary(corpus.dictionary)
        self.model.save(f"glove_{execution_id}.model")

        recipe_vectors: pd.Series = self.recipes.apply(self.recipe_vectorizer)

        np.save(f"recipeglove_vectors{execution_id}.npy", recipe_vectors.values)

        for vec_index, vec in self.recipes.items():
            self.index.add_item(vec_index, vec)

        out_path = f"recipeglove_{execution_id}.ann"

        self.index.build(10)
        self.index.save(out_path)

        return out_path

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients in a recipe to the average of each ingredient's embedding vector.

        """

        ingredient_vecs = np.array([self.model.wv[ingredient] for ingredient in recipe])

        recipe_vec = np.mean(ingredient_vecs, axis=0)

        return recipe_vec
