# Import necessary libraries
# pip install scikit-learn
# pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

# Generate dataset
X, Y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)

# Scale the dataset
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with the appropriate loss function for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=500)

# Generate new data for prediction
X_new, Y_real = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
X_new = scaler.transform(X_new)

# Predict the class and probability for new data
Y_prob = model.predict(X_new)
Y_class = (Y_prob > 0.5).astype("int32")

# Print the results
for i in range(len(X_new)):
    print(f"X={X_new[i]}, Predicted_probability={Y_prob[i]}, Predicted_class={Y_class[i]}")
