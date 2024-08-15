from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt

# Download MNIST data and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Plot the first image in the dataset
plt.imshow(X_train[0])  # Added cmap='gray' for better visualization
plt.show()

# Print the shape of the first image
print(X_train[0].shape)

# Reshape data to fit the model
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# One-hot encode target column
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

# Print the first one-hot encoded label
print(Y_train[0])

# Create model
model = Sequential()

# Add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)

# Predict the first 4 images in the test set
predictions = model.predict(X_test[:4])
print(predictions)

# Actual results for the first 4 images in the test set
print(Y_test[:4])