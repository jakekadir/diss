import ast
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def parse_string_as_tuple(tuple_string: str) -> List[str]:
    """
    Takes a strings and safely attempts to parse it as Python code.
    """

    try:

        parsed: Tuple[str] = ast.literal_eval(tuple_string)

        # was not parsed as desired, so reject
        if type(parsed) is not tuple:

            return tuple_string

        else:

            return parsed

    except Exception:

        return tuple_string


def check_is_dir(path: Path) -> Path:

    if path.exists() and path.is_dir():

        return path

    else:
        raise ValueError(
            f"The provided directory path, {path}, doesn't exist or is not a directory."
        )


def check_file_exists(path: Path) -> Path:

    if path.exists():

        return path

    else:
        raise ValueError(f"No file exists at the given path, {path}")


def get_recipes(path: Path) -> pd.DataFrame:
    """
    Opens the recipe dataset at the specified path, converting the nested fields to array-like objects rather than strings.

    Inputs:
        path: pathlib.Path, the path to the recipe dataset
    Outputs:
        pd.DataFrame, the prepared DataFrame of recipes
    """

    # ingest data
    recipes: pd.DataFrame = pd.read_csv(path)

    # format the ingredients
    recipes = recipes.drop(
        recipes[recipes["RecipeIngredientParts"].str[:2] != "c("].index
    )
    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].str[1:]

    # parse the string as a tuple
    recipes["RecipeIngredientParts"] = recipes["RecipeIngredientParts"].apply(
        parse_string_as_tuple
    )

    # reset index for consistency
    recipes = recipes.reset_index(drop=True)

    return recipes
