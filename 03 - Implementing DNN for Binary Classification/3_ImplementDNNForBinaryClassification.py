# pip install tensorflow
# pip install pandas
# pip install scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
data = pd.read_csv('C:/Users/vivek/PycharmProjects/Prac3/diabetes.csv', delimiter=',')

# Split into input (X) and output (Y) variables
X = data.iloc[:, 0:8].values
Y = data.iloc[:, 8].values

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model
model = Sequential()
model.add(Dense(units=12, activation='relu', input_dim=8)) # Add input layer and first hidden layer
model.add(Dense(units=8, activation='relu')) # Add second hidden layer
model.add(Dense(units=1, activation='sigmoid')) # Add output layer

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X_train, Y_train, epochs=150, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

# Predict the output for the training data
prediction = model.predict(X, batch_size=4)

# Print the predictions
print(prediction)

exec("for i in range(5):print(X[i].tolist(),prediction[i], Y[i])")
