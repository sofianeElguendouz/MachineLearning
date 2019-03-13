##################################################
# Specifying a model with keras
##################################################

# Import necessary modules
import keras
from keras.layers import Dense
from keras.models import Sequential

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(ncols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))
