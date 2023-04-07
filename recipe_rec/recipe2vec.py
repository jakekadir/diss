from gensim.models import Word2Vec, KeyedVectors
import logging  # Setting up the loggings to monitor gensim
from pathlib import Path
from annoy import AnnoyIndex
from data_loader import get_recipes
from typing import List
import numpy as np
from recommender_system import IngredientRecommender
import pandas as pd

class Recipe2Vec(IngredientRecommender):
    
    def __init__(self, dataset_path: Path, verbose: bool, execution_id: str):
        # set parent attributes
        super().__init__("Recipe2Vec", "", 100)
        
        # generate the index
        self.generate_recipe2vec_index(dataset_path, verbose, execution_id)


    def generate_recipe2vec_index(self,
                                  dataset_path: Path,
                                  verbose: bool,
                                  execution_id: str) -> AnnoyIndex:
    
        """
        Takes a path to a dataset, loads the data and produces recipe2vec embeddings
        for a given column of the dataset.
        
        An Annoy Index is constructed for these embeddings and written to a file
        which incorporates the execution_id in the filename.
        
        """
        
        if verbose:
            logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
        
        # load data
        self.recipes = get_recipes(dataset_path).reset_index(drop=True)
        
        vec_size = 100
        
        # create model
        self.model = Word2Vec(
            self.recipes["RecipeIngredientParts"].values,
            # use skipgram, not CBOW
            sg=1,
            vector_size=vec_size,
            # ensures rarely-occurring ingredients still are given a vector
            min_count=1,
            epochs=30
        )
        
        # train the model
        # model.train(recipes, total_examples=model.corpus_count, epochs=30, report_delay=1)
        
        # build vocab??
        # model.build_vocab(recipes, progress_per=10000)
        
        # map the recipes to vectors
        recipe_vectors: pd.Series = self.recipes["RecipeIngredientParts"].apply(self.recipe_vectorizer)   

        # saves the vectors to allow the index to be re-built
        np.save("recipe2vec_vectors_{execution_id}.npy",recipe_vectors.values)

        # build an index
        self.index = AnnoyIndex(vec_size, "angular")

        # populate
        for vec_index,vec in recipe_vectors.items():
            self.index.add_item(vec_index, vec)
            
        # build and save
        out_path = f"recipe2vec_{execution_id}.ann"
        
        self.index.build(10)
        self.index.save(out_path)
        
        return out_path

    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients in a recipe to the average of each ingredient's embedding vector.
        
        """
        
        ingredient_vecs = np.array([self.model.wv[ingredient] for ingredient in recipe])

        recipe_vec = np.mean(ingredient_vecs,axis=0)
        
        return recipe_vec
    
    # def get_recommendations(self, recipe: List[str], n_recommendations: int) -> pd.DataFrame:
    #     """
    #     Creates a recipe vector from a list of ingredients and queries the Annoy index for the `n_recommendations` nearest neighbours.
        
    #     Inputs:
    #         - `recipe`: `List[str]`, a list of string ingredients
    #         - `n_recommendations`: `int`, the number of recommendations to return
    #     Outputs:
    #         - `pd.DataFrame`, a sorted DataFrame of the recommended recipes
    #     """
        
    #     try:
        
    #         # get the vector of the recipe
    #         recipe_vec = self.recipe_vectorizer(recipe)
        
    #         # get closest vectors from the dataset
    #         rec_indexes = self.index.get_nns_by_vector(recipe_vec, n_recommendations)
            
    #         # translate recommendations into recipes
    #         recs = self.recipes.iloc[rec_indexes]
        
    #         return recs
        
    #     except KeyError:
    #         raise ValueError("One of the given ingredients did not exist in the training dataset.")