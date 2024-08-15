# pip install numpy
# pip install tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

# Define the model
model = Sequential()
model.add(Input(shape=(2,)))  # Define input layer
model.add(Dense(units=2, activation='relu'))  # Hidden layer with 2 neurons
model.add(Dense(units=1, activation='sigmoid'))  # Output layer with 1 neuron

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Train the model
model.fit(X, Y, epochs=1000, batch_size=4)

# Evaluate the model
loss, accuracy = model.evaluate(X, Y)
print(f'Accuracy: {accuracy * 100}%')

# Make predictions
predictions = model.predict(X)
print('Predictions:')
print(predictions)