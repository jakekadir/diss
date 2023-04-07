import pandas as pd
import pathlib
from data_loader import get_recipes
from sentence_transformers import SentenceTransformer
from annoy import AnnoyIndex
from pathlib import Path
from recommender_system import IngredientRecommender
from typing import List
import numpy as np

class SBERTRecommender(IngredientRecommender):
    
    def __init__(self,
                 dataset_path: Path,
                 col_to_embed: str,
                 execution_id: str):
        super().__init__("SBERT", "", 384)
        
        self.generate_sbert_index(dataset_path,
                                  col_to_embed,
                                  execution_id)
        
    def recipe_vectorizer(self, recipe: List[str]) -> np.ndarray:
        """
        Maps a list of ingredients to a vector.
        
        """
        
        # combine ingredients into a string
        joined_ingredients = ",".join(recipe)
        
        # retrieve BERT vector for string        
        recipe_vec = self.model.encode(joined_ingredients)
        
        return recipe_vec
        
    def generate_sbert_index(self, 
                             dataset_path: Path,
                             col_to_embed: str,
                             execution_id: str) -> pathlib.Path:
        
        """
        Takes a path to a dataset, loads the data and produces sentence-BERT embeddings
        for a given column of the dataset.
        
        An Annoy Index is constructed for these embeddings and written to a file
        which incorporates the execution_id in the filename.
        
        """
        
        # load the transformer model
        transformer_model: str = "paraphrase-MiniLM-L6-v2"
        self.model: SentenceTransformer = SentenceTransformer(transformer_model)

        self.recipes: pd.DataFrame = get_recipes(dataset_path).reset_index(drop=True)

        # generate embeddings
        embeddings = self.model.encode(self.recipes[col_to_embed].values)
        embedding_indexes = zip(self.recipes[col_to_embed].index, embeddings)
        
        self.index = AnnoyIndex(self.vec_size, "angular")

        for i in range(len(embeddings)):
            self.index.add_item(i, embeddings[i])
        # for embed in embedding_indexes:
        #     self.index.add_item(embed[0], embed[1])
            
        out_path = f"sbert_{execution_id}.ann"
        
        self.index.build(10)
        self.index.save(out_path)
        
        return out_path
    
    # def get_recommendations(self, recipe: List[str], n_recommendations: int) -> pd.DataFrame:
    #     """
    #     Creates a recipe vector from a list of ingredients and queries the Annoy index for the `n_recommendations` nearest neighbours.
        
    #     Inputs:
    #         - `recipe`: `List[str]`, a list of string ingredients
    #         - `n_recommendations`: `int`, the number of recommendations to return
    #     Outputs:
    #         - `pd.DataFrame`, a sorted DataFrame of the recommended recipes
    #     """
        
    #     # get the vector of the recipe
    #     recipe_vec = self.recipe_vectorizer(recipe)
    
    #     # get closest vectors from the dataset
    #     rec_indexes = self.index.get_nns_by_vector(recipe_vec, n_recommendations)
        
    #     # translate recommendations into recipes
    #     recs = self.recipes.iloc[rec_indexes]
    
    #     return recs
        

