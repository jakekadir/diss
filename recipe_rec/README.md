# `recipe_rec`

`recipe_rec` is a Python package implementing four distinct recommender systems designed to deliver recommendations for the recipe dataset found at `/data/recipes.csv`.

The six systems implemented are:
- `recipe2vec`, a custom-trained Word2Vec language model using the recipe' ingredients as tokens.
- `fastRecipe`, a custom-trained fastText language model using the recipes' ingredients as tokens.
- `sbert`, a system using pre-trained SBERT embeddings to produce a recipe's embedding from a list of ingredients.
- `feature_generation`, a system using classifiers (which receive an input of a pre-trained SBERT embedding) to label semantic features of the recipe (e.g. savoury, spicy etc.) to produce a low-dimensional vector space.
- `simple_search`, a basic system that returns recipes that have the highest number of ingredients that match those given in the query. 
- `TF-IDF`, a system using TF-IDF vectors of the recipes' ingredients.