# Import necessary libraries
# pip install tensorflow
# pip install matplotlib
# pip install -U scikit-learn
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1_l2

# Generate a dataset with a moon shape
X, Y = make_moons(n_samples=100, noise=0.2, random_state=1)

# Split the dataset into training and testing sets
n_train = 30
trainX, testX = X[:n_train, :], X[n_train:, :]
trainY, testY = Y[:n_train], Y[n_train:]

# Define the model
model = Sequential()
model.add(Dense(500, input_dim=2, activation='relu', kernel_regularizer=l1_l2(l1=0.001, l2=0.001)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100)

# Plot the training and validation accuracy
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()