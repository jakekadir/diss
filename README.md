This repository is the codebase for Jake Kadir's dissertation project.

The repository includes three primary directories:
- `recipe_rec` - a Python package that implements four recommender systems to deliver recommendations for a dataset
- `data` - includes the primary data source, `recipes.csv` and holds intermediate binary files used by the recommender systems
- `notebooks` - includes Jupyter Notebooks used to perform EDA on the dataset

To run the `recipe_rec` package, import the `load_dataset` function from `recipe_rec.data` and pass a path to the `recipes.csv` file (part of the dataset uploaded at https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews). Various recommender systems can be imported from `recipe_rec.models`.