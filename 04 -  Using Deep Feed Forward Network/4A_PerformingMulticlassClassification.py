# pip install scikit-learn
# pip install tensorflow

from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic dataset
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Scale the dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))  # First hidden layer with 4 neurons
model.add(Dense(4, activation='relu'))  # Second hidden layer with 4 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer with 1 neuron

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam')

# Fit the model
model.fit(X, Y, epochs=500)

# Generate new samples
Xnew, Yreal = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scaler.transform(Xnew)

# Predict the class for new samples
Ynew = (model.predict(Xnew) > 0.5).astype("int32")

# Print the results
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s, Desired=%s" % (Xnew[i], Ynew[i][0], Yreal[i]))