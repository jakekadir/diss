import logging
from typing import Dict

import pandas as pd
import tqdm

from recipe_rec import RANDOM_STATE
from recipe_rec.utilities import recipes
from recipe_rec.recommender_system import RecommenderSystem

logger = logging.getLogger(__name__)


def generate_test_data(
    rec_systems: Dict[str, RecommenderSystem],
    n_recipes: int,
    n_recommendations: int,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Generates test data for a given dictionary of recommender systems.

    Parameters:
        - `rec_systems: Dict[str, RecommenderSystem]`: a dictionary of recommender systems (values) and string labels (keys).
        - `n_recipes: int`: the number of base recipes to sample.
        - `n_recommendations: int`: the number of recommendations to retrieve for each recipe.
        - `verbose: bool = False`:
    Returns:
        - `pd.DataFrame`: a DataFrame of the sampled recipes and the recommendations made by each recommender system.

    """
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

    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL)

    logger.info("Choosing sample of recipes to act as recommendation queries.")

    # choose a sample of recipes to get recommendations for, fix across all systems
    sample: pd.DataFrame = recipes.sample(n=n_recipes, random_state=RANDOM_STATE)

    # for each recommender
    for system in (pbar := tqdm(rec_systems, disable=not verbose)):
        if verbose:
            pbar.set_description(f"Generating recommendations using {system} system.")

        for recipe_index in sample.index:

            # get recommendations
            recommendations: pd.DataFrame = rec_systems[system].get_recommendations(
                recipe=recipes.loc[recipe_index]["RecipeIngredientParts"],
                n_recommendations=n_recommendations,
                search_id=recipe_index,
                get_recipes=True,
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
