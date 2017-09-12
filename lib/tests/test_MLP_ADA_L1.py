import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from lib.multilayer_perceptron import MLP

layer_shape = np.array([[30, 784], [15, 30], [10, 15]])
num_features = 784
num_labels = 10

nn1 = MLP(num_features, num_labels, layer_shape,
          optimizer_type='adam', regularization_type='l1')
nn1.print_fields()

z = nn1.forward_prop()
c = nn1.compute_cost_fn(z)
optimizer = nn1.create_optimizer(c)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Model expects that each column is one sample, so each row is a feature
train_images = mnist.train.images.T
train_labels = mnist.train.labels.T

costs = nn1.train(train_images, train_labels, batch_size=64)

plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.show()