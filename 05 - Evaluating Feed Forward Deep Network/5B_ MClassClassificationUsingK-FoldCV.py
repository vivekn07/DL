# Loading necessary libraries
# pip install tensorflow
# pip install scikeras

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Loading dataset
# Tell pandas to treat the first row as a header
df = pd.read_csv('C:/Users/vivek/PycharmProjects/Prac5/flowers.csv', header=0)
print(df)

# Splitting dataset into input and output variables
# Adjust column indexing to start from 0
X = df.iloc[:, 0:4].astype(float)
y = df.iloc[:, 4]
print(X)
print(y)

# Encoding string output into numeric output
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print(encoded_y)

# Use to_categorical from tensorflow.keras.utils
dummy_Y = to_categorical(encoded_y)
print(dummy_Y)

# Define the baseline model
def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create the model
estimator = baseline_model()

# Fit the model
estimator.fit(X, dummy_Y, epochs=100, shuffle=True)

# Predict using the model
action = estimator.predict(X)

# Print the first 25 actual and predicted values
for i in range(25):
    print(dummy_Y[i])
print('================================')
for i in range(25):
    print(action[i])
