import numpy as np
import tensorflow as tf
import math

class MLP:
    '''
    num_layer: number of layers
        given num_layer=L, layers is indexed l(0), l(1), ..., l(L-1)
    layer_shape: shape of each layer
        a (num_layer, 2)
    hidden_activation: activation function used in each layer
        defaults to using ReLU for layers l(0) to l(L-2)
        input: a dictionary for mapping layer number to activation
            {1: 'relu', 2: 'sigmoid'}
    output_activation: activation for final layer
        defaults to using softmax
        input: string
    weight_init: weight intialization for each layer
    regularization: type of regularization
    optimizer: method for backprop
    '''
    num_features = None
    num_labels = None

    num_layers = None
    layer_shape = None

    learning_rate = None
    reg_lambda = None
    beta_1 = None
    beta_2 = None
    epsilon = None

    hidden_activation = None
    output_activation = None
    weight_init_type = None
    regularization_type = None
    optimizer_type = None

    features = None
    labels = None

    optimizer = None
    cost_fn = None

    # Graph is needed to instance the graph properly (i.e. using another model
    # somewhere else will effect this model)
    __graph = None
    __parameters = None
    __HIDDEN_ACTIVATION = ['relu', 'softmax', 'sigmoid']
    __OUTPUT_ACTIVATION = ['softmax', 'sigmoid']
    __WEIGHT_INIT = ['xavier']
    __REGULARIZER = ['l2', 'l1']
    __OPTIMIZER = ['gradient', 'momentum', 'rmsprop', 'adam']

    def __init__(self, num_features, num_labels, layer_shape,
                 learning_rate=0.01, reg_lambda=0.1,
                 beta_1=0.9, beta_2=0.999, epsilon=10e-8,
                 hidden_activation=None, output_activation=None,
                 weight_init_type=None, regularization_type=None,
                 optimizer_type=None):

        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.num_features = num_features
            self.num_labels = num_labels

            # Network architecture
            self.layer_shape = layer_shape
            self.num_layers = layer_shape.shape[0]

            # Hyperparameters
            self.learning_rate = learning_rate
            self.reg_lambda = reg_lambda
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.epsilon = epsilon

            assert(self.layer_shape[0][1] == self.num_features)
            assert(self.layer_shape[self.num_layers - 1][0] == self.num_labels)

            self.features = tf.placeholder(dtype=tf.float32,
                                           shape=[self.num_features, None])
            self.labels = tf.placeholder(dtype=tf.float32,
                                         shape=[self.num_labels, None])

            if hidden_activation is None:
                self.hidden_activation = [self.__HIDDEN_ACTIVATION[0] for _ in range(self.num_layers - 1)]
            else:
                self.hidden_activation = []
                for i in range(self.num_layers-1):
                    if i in hidden_activation:
                        if hidden_activation[i] in self.__HIDDEN_ACTIVATION:
                            self.hidden_activation.append(hidden_activation[i])
                    elif i != self.num_layers-1:
                        self.hidden_activation.append(self.__HIDDEN_ACTIVATION[0])

            assert(len(self.hidden_activation) == self.num_layers - 1)

            if output_activation is None:
                self.output_activation = self.__OUTPUT_ACTIVATION[0]
            else:
                if output_activation in self.__OUTPUT_ACTIVATION:
                    self.output_activation = output_activation

            assert(self.output_activation is not None)

            if weight_init_type is None:
                self.weight_init_type = self.__WEIGHT_INIT[0]
            else:
                if weight_init_type in self.__WEIGHT_INIT:
                    self.weight_init_type = weight_init_type

            assert(self.weight_init_type is not None)

            if regularization_type is None:
                self.regularization_type = self.__REGULARIZER[0]
            else:
                if regularization_type in self.__REGULARIZER:
                    self.regularization_type = regularization_type

            assert(self.regularization_type is not None)

            if optimizer_type is None:
                self.optimizer_type = self.__OPTIMIZER[0]
            else:
                if optimizer_type in self.__OPTIMIZER:
                    self.optimizer_type = optimizer_type

            assert(self.optimizer_type is not None)

            self.__parameters = self.__init_parameters()

        return

    def __init_parameters(self):
        parameters = {}
        for i in range(self.num_layers):
            layer_shape = self.layer_shape[i]
            if i != 0:
                prev_shape = self.layer_shape[i-1]
                assert(prev_shape[0] == layer_shape[1])

            weight = tf.get_variable('W' + str(i), shape=[layer_shape[0], layer_shape[1]],
                                     initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable('b' + str(i), shape=[layer_shape[0], 1],
                                   initializer=tf.zeros_initializer())
            parameters['W' + str(i)] = weight
            parameters['b' + str(i)] = bias
        return parameters

    def forward_prop(self):
        with self.__graph.as_default():
            a = self.features

            for i in range(self.num_layers):
                z = tf.matmul(self.__parameters['W' + str(i)],
                              a) + self.__parameters['b' + str(i)]
                # output layer needs to compare with labels
                if i == self.num_layers-1:
                    break
                if self.hidden_activation[i] == self.__HIDDEN_ACTIVATION[0]:
                    a = tf.nn.relu(z)
                elif self.hidden_activation[i] == self.__HIDDEN_ACTIVATION[1]:
                    a = tf.nn.softmax(z)
                elif self.hidden_activation[i] == self.__HIDDEN_ACTIVATION[2]:
                    a = tf.nn.sigmoid(z)

        return z

    def compute_cost_fn(self, z):
        with self.__graph.as_default():
            logits = tf.transpose(z)
            labels = tf.transpose(self.labels)
            cost_fn = None
            if self.output_activation == self.__OUTPUT_ACTIVATION[0]:
                cost_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=labels
                ))
            elif self.output_activation == self.__OUTPUT_ACTIVATION[1]:
                cost_fn = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels
                ))

            regularizer = tf.Variable(0, dtype=tf.float32)
            if self.regularization_type == self.__REGULARIZER[0]:
                for i in range(self.num_layers):
                    regularizer = tf.add(tf.nn.l2_loss(self.__parameters['W' + str(i)]),
                                         regularizer)
            elif self.regularization_type == self.__REGULARIZER[1]:
                for i in range(self.num_layers):
                    regularizer = tf.add(tf.norm(self.__parameters['W' + str(i)],
                                         ord=1), regularizer)

            cost_fn = tf.add(cost_fn, self.reg_lambda * regularizer)
            self.cost_fn = cost_fn
            assert(self.cost_fn is not None)

        return cost_fn

    def create_optimizer(self, cost_fn=None):
        with self.__graph.as_default():
            if cost_fn is None:
                cost_fn = self.cost_fn

            optimizer = None
            if self.optimizer_type == self.__OPTIMIZER[0]:
                optimizer = tf.train.GradientDescentOptimizer(
                    learning_rate=self.learning_rate).minimize(cost_fn)
            elif self.optimizer_type == self.__OPTIMIZER[1]:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                       momentum=self.beta_1).minimize(cost_fn)
            elif self.optimizer_type == self.__OPTIMIZER[2]:
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                                      momentum=self.beta_2).minimize(cost_fn)
            elif self.optimizer_type == self.__OPTIMIZER[3]:
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                   beta1=self.beta_1,
                                                   beta2=self.beta_2).minimize(cost_fn)

            self.optimizer = optimizer
            assert(self.optimizer is not None)

        return optimizer

    def train(self, x_train, y_train, num_epochs=50, batch_size=64):
        assert(x_train.shape[1] == y_train.shape[1])
        m = x_train.shape[1]
        num_batches = int(m / batch_size)
        costs = []
        with tf.Session(graph=self.__graph) as sess:
            sess.run(tf.global_variables_initializer())
            print('Start Training...')
            # An epoch is one whole pass through the entire data
            for epoch in range(num_epochs):
                print(epoch)
                epoch_cost = 0
                mini_batches = self.create_mini_batches(x_train, y_train, batch_size)

                for mini_batch in mini_batches:
                    (mb_x, mb_y) = mini_batch
                    val, mb_cost = sess.run([self.optimizer, self.cost_fn], feed_dict={
                        self.features: mb_x,
                        self.labels: mb_y
                    })
                epoch_cost += mb_cost / num_batches

                if epoch % 5 == 0:
                    costs.append(epoch_cost)
                if epoch % 10 == 0:
                    print('Cost after epoch %i: %f' % (epoch, epoch_cost))

        return costs

    @staticmethod
    def create_mini_batches(x, y, batch_size):
        assert(x.shape[1] == y.shape[1])
        m = x.shape[1]
        permutation = list(np.random.permutation(m))
        shuffled_x = x[:, permutation]
        shuffled_y = y[:, permutation]

        mini_batches = []
        num_complete_mini_batches = int(math.floor(m / batch_size))
        for i in range(num_complete_mini_batches):
            mb_x = shuffled_x[:, i * batch_size:(i+1) * batch_size]
            mb_y = shuffled_y[:, i * batch_size:(i+1) * batch_size]
            mb = (mb_x, mb_y)
            mini_batches.append(mb)

        if m % batch_size != 0:
            mb_x = shuffled_x[:, num_complete_mini_batches * batch_size:]
            mb_y = shuffled_y[:, num_complete_mini_batches * batch_size:]
            mb = (mb_x, mb_y)
            mini_batches.append(mb)

        return mini_batches

    def print_fields(self):
        print("Number of layers: ", self.num_layers)
        print("Number of features: ", self.num_features)
        print("Number of labels: ", self.num_labels)
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                print("Layer: ", i, " (", self.layer_shape[i][0],", ",
                      self.layer_shape[i][1], ") ", self.hidden_activation[i])
            else:
                print("Layer: ", i, " (", self.layer_shape[i][0], ", ",
                      self.layer_shape[i][1], ") ", self.output_activation)

        print("Learning rate: ", self.learning_rate)
        print("Weight initializer: ", self.weight_init_type)
        print("Regularization: ", self.regularization_type)
        print("Optimizer: ", self.optimizer_type)
