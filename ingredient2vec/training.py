import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Dot
from keras.layers.core import Dense, Reshape
import keras

import pandas as pd
import numpy as np
import ast
import tqdm
import pickle

RANDOM_SEED = 42
AUTOTUNE = tf.data.AUTOTUNE
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

vocab = np.load("./tokenised/vocab.pkl", allow_pickle=True)

vocab_size = len(vocab)


# X_targets = [
#     target_id for recipe in X for target_id in recipe[0]
# ]

# X_contexts = [
#     target_id for recipe in X for target_id in recipe[1]
# ]
# X = np.array([X_targets, X_contexts])

# Y = np.array([
#     label for recipe in Y for label in recipe
# ])
def training_generator():
    X = np.load("./tokenised/X.pkl", allow_pickle=True)
    Y = np.load("./tokenised/Y.pkl", allow_pickle=True)

    for recipe_index in range(len(X)):
        
        yield X[recipe_index], Y[recipe_index]
        # for ingredient_pair_index in range(len(X[recipe_index])):
        #     record = [
        #             X[recipe_index][0][ingredient_pair_index],
        #             X[recipe_index][1][ingredient_pair_index]
        #         ] 
            
        #     print(record)
            
        #     yield record, Y[recipe_index][ingredient_pair_index] 


# define vector size for embeddings
embedding_size = 100

target_inputs = keras.Input(shape=(1,))

target_x = Embedding(
    # size of input vector - equal to vocab size
    input_dim=vocab_size,
    output_dim=embedding_size,
    # distribution to sample random values from for initial embeddings
    embeddings_initializer="glorot_uniform",
    input_length=1
)(target_inputs)
target_output = Reshape((embedding_size, ))(target_x)

target_model = keras.Model(inputs=target_inputs, outputs=target_output)

context_inputs = keras.Input(shape=(1,))

context_x = Embedding(
    # size of input vector - equal to vocab size
    input_dim=vocab_size,
    output_dim=embedding_size,
    # distribution to sample random values from for initial embeddings
    embeddings_initializer="glorot_uniform",
    input_length=1
)(context_inputs)
context_output = Reshape((embedding_size, ))(context_x)

context_model = keras.Model(inputs=context_inputs, outputs=context_output)

# take the dot product of the outputs of the target and context models
dot_layer = Dot(axes=1, normalize=False)([target_model.output, context_model.output])

# pass the dot product to a dense layer
combined_out = Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid")(dot_layer)

# compile to a model
combined_model = keras.Model(inputs=[target_model.input, context_model.input], outputs=combined_out)

print(combined_model.summary())

combined_model.compile(loss="categorical_crossentropy",optimizer="adam")

print("About to fit:")

trainer = training_generator()

combined_model.fit(
    trainer,
    verbose=2
)

ingredient_layer = combined_model.layers[3].get_weights()

np.save("weights.npy",ingredient_layer)