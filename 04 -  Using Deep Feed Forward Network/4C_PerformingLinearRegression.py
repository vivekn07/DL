# Import necessary libraries
# pip install scikit-learn
# pip install tensorflow
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler

# Generate a regression dataset
X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# Initialize the MinMaxScaler for both features and target
scalerX, scalerY = MinMaxScaler(), MinMaxScaler()

# Fit the scaler on the dataset
scalerX.fit(X)
scalerY.fit(Y.reshape(100, 1))

# Transform the dataset
X = scalerX.transform(X)
Y = scalerY.transform(Y.reshape(100, 1))

# Define the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Fit the model
model.fit(X, Y, epochs=1000, verbose=0)

# Generate new data for prediction
Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.1, random_state=1)

# Transform the new data
Xnew = scalerX.transform(Xnew)

# Predict the output
Ynew = model.predict(Xnew)

# Print the predictions
for i in range(len(Xnew)):
    print("X=%s, Predicted=%s" % (Xnew[i], Ynew[i]))