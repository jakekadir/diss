from typing import List
from pathlib import Path
from data_loader import import_recipes
from recommender_system import RecommenderSystem
from annoy import AnnoyIndex
import pandas as pd

RANDOM_STATE = 42

def generate_test_data(rec_systems: List[RecommenderSystem],
                       dataset_path: Path, 
                       num_recipes: int, 
                       num_recommendations: int,
                       out_path: Path):
    
    # to store evaluation data
    evaluation_data = pd.DataFrame({
        "Rec_Name" : [],
        "Rec_Description" : [],
        "Rec_Ingredients" : [],
        "Rec_System" : [],
        "Origin_Name" : [],
        "Origin_Description" : [],
        "Origin_Ingredients" : []
    })
    
    # load dataset
    recipes = import_recipes(dataset_path)
    
    # choose a sample of recipes to get recommendations for, fix across all systems
    sample = recipes.sample(n=num_recipes, random_state=RANDOM_STATE)
    
    # for each recommender
    for system in rec_systems:
        
            index = AnnoyIndex(system.vec_size, "angular")
            index.load(system.index_path)
            
            for recipe_index in sample.index.values:
                
                # get the IDs of the recommendations
                recommendation_ids = index.get_nns_by_item(recipe_index, num_recommendations)
                
                # get the full records for the recommendations 
                recommendations = recipes.iloc[recommendation_ids]
                
                # note which system gave these recommendations
                recommendations["Rec_System"] = system.name
                
                # rename columns
                recommendations = recommendations.rename({
                    "Name": "Rec_Name",
                    "Description" : "Rec_Description",
                    "Ingredients": "Rec_Ingredients",
                })
                
                # drop unneeded columns
                recommendations = recommendations[["Rec_Name","Rec_Description","Rec_Ingredients","Rec_System"]]
                
                # get the details of the originating recipe
                origin_recipe = sample.iloc[recipe_index]
                
                recommendations["Origin_Name"] = origin_recipe["Name"]
                recommendations["Origin_Description"] = origin_recipe["Description"]
                recommendations["Origin_Ingredients"] = origin_recipe["Ingredients"]
                
                # add to the master df
                pd.concat([evaluation_data, recommendations])
    
    evaluation_data.to_csv(out_path)
                