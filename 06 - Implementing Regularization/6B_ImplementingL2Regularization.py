# Import necessary libraries
# pip install tensorflow
# pip install matplotlib
# pip install -U scikit-learn
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.regularizers import l2

# Generate a simple binary classification dataset
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:]
trainY, testY = Y[:n_train], Y[n_train:]

# Define the model with L2 regularization
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu',kernel_regularizer=l2(0.001)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the training data and validate on the testing data
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=4000)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()