from pathlib import Path
from typing import Dict

import pandas as pd
from annoy import AnnoyIndex

from recipe_rec import RANDOM_STATE, recipes
from recipe_rec.data_loader import get_recipes
from recipe_rec.recommender_system import RecommenderSystem


def generate_test_data(
    rec_systems: Dict[str, RecommenderSystem],
    num_recipes: int,
    n_recommendations: int,
):
    # to store evaluation data
    evaluation_data = pd.DataFrame(
        {
            "Rec_Name": [],
            "Rec_Description": [],
            "Rec_Ingredients": [],
            "Rec_System": [],
            "Recommendation_Id": [],
            "Rank": [],
            "Origin_Name": [],
            "Origin_Description": [],
            "Origin_Ingredients": [],
        }
    )

    # choose a sample of recipes to get recommendations for, fix across all systems
    sample = recipes.sample(n=num_recipes, random_state=RANDOM_STATE)

    # for each recommender
    for system in rec_systems:

        for recipe_index in sample.index:

            # get recommendations
            recommendations = rec_systems[system].get_recommendations(
                recipe=recipes.loc[recipe_index]["RecipeIngredientParts"],
                n_recommendations=n_recommendations,
                search_id=recipe_index,
            )

            # drop the ID column for the recommendations to grab the recipe's IDs
            recommendations.reset_index(inplace=True, names=["Recommendation_Id"])

            # extract the new index to use as the rank of the recommendations
            recommendations.reset_index(inplace=True, names=["Rank"])

            # note which system gave these recommendations
            recommendations["Rec_System"] = system

            # rename columns
            recommendations.rename(
                {
                    "Name": "Rec_Name",
                    "Description": "Rec_Description",
                    "RecipeIngredientParts": "Rec_Ingredients",
                },
                axis=1,
                inplace=True,
            )

            # drop unneeded columns
            recommendations = recommendations[
                [
                    "Rec_Name",
                    "Rec_Description",
                    "Rec_Ingredients",
                    "Rec_System",
                    "Recommendation_Id",
                    "Rank",
                ]
            ]

            # get the details of the originating recipe

            recommendations["Origin_Name"] = [recipes.loc[recipe_index]["Name"]] * len(
                recommendations.index
            )
            recommendations["Origin_Id"] = [recipe_index] * len(recommendations.index)
            recommendations["Origin_Description"] = [
                recipes.loc[recipe_index]["Description"]
            ] * len(recommendations.index)
            recommendations["Origin_Ingredients"] = [
                recipes.loc[recipe_index]["RecipeIngredientParts"]
            ] * len(recommendations.index)

            # add to the master df
            evaluation_data = pd.concat([evaluation_data, recommendations])

    return evaluation_data