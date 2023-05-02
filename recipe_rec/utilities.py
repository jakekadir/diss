from pathlib import Path

import pandas as pd

from recipe_rec.data import store

# replaces spaces in a list of strings with "_"; used in preprocessing
space_replacer = lambda recipe: [ingredient.replace(" ", "_") for ingredient in recipe]


def check_is_dir(path: Path) -> Path:

    if path.exists() and path.is_dir():

        return path

    else:
        raise ValueError(
            f"The provided directory path, {path}, doesn't exist or is not a directory."
        )


def check_file_exists(path: Path) -> Path:

    if isinstance(path, Path):

        if path.exists():

            return path

        else:
            raise FileNotFoundError(f"No file exists at the given path {path}")
    else:
        raise ValueError(
            f"The given argument has type {type(path)}; is should have type pathlib.Path"
        )


def check_dataset_loaded():
    err = RuntimeError(
        "The dataset has not been imported. Call the load_datastet() function from recipe_rec.data before instantiating an object."
    )
    if "recipes" not in store.keys():
        raise err
    elif store["recipes"] is None or type(store["recipes"]) is not pd.DataFrame:

        raise err


def get_dataset():

    check_dataset_loaded()

    return store["recipes"]


def get_ingredients():

    check_dataset_loaded()

    return store["unique_ingredients"]
