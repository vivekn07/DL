# pip install tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# Print a message indicating the start of the demo
print("======== Matrix Multiplication ========\n")

# Define a constant tensor x with shape [2, 3]
x = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print("Matrix X:\n{}\n".format(x))

# Define a constant tensor y with shape [3, 2]
y = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
print("Matrix Y:\n{}\n".format(y))

# Perform matrix multiplication of x and y
z = tf.matmul(x, y)
print("Product of X & Y:\n{}\n".format(z))

print("======================================\n")

# Generate a random matrix A with values between 3 and 10 and shape [2, 2]
matrix_A = tf.random.uniform([2, 2], minval=3, maxval=10)
print("Matrix A:\n{}\n".format(matrix_A))

# Compute the eigenvalues and eigenvectors of matrix A
eigen_values_A, eigen_vectors_A = tf.linalg.eigh(matrix_A)
print("Eigen Vectors:\n{}\n\nEigen Values:\n{}\n".format(eigen_vectors_A, eigen_values_A))