import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

n1 = 16
n2 = 32
n3 = 64
ksize = 5
learning_rate = 0.001
batch_size = 128
n_epochs = 5

weights = {
    'ce1': tf.Variable(tf.random_normal([ksize, ksize, 1, n1], stddev=0.1)),
    'ce2': tf.Variable(tf.random_normal([ksize, ksize, n1, n2], stddev=0.1)),
    'ce3': tf.Variable(tf.random_normal([ksize, ksize, n2, n3], stddev=0.1)),
    'cd3': tf.Variable(tf.random_normal([ksize, ksize, n2, n3], stddev=0.1)),
    'cd2': tf.Variable(tf.random_normal([ksize, ksize, n1, n2], stddev=0.1)),
    'cd1': tf.Variable(tf.random_normal([ksize, ksize, 1, n1],  stddev=0.1))
}
biases = {
    'be1': tf.Variable(tf.random_normal([n1], stddev=0.1)),
    'be2': tf.Variable(tf.random_normal([n2], stddev=0.1)),
    'be3': tf.Variable(tf.random_normal([n3], stddev=0.1)),
    'bd3': tf.Variable(tf.random_normal([n2], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n1], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1],  stddev=0.1))
}


def CNNAutoEncoder(X, W, b, keepprob):
    image = tf.reshape(X, shape=[-1, 28, 28, 1])

    # Encoder
    ce1 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d(image,
                                                          W['ce1'],
                                                          strides=[1, 2, 2, 1],
                                                          padding='SAME'),
                                             b['be1'])),
                        keepprob)

    ce2 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d(ce1,
                                                          W['ce2'],
                                                          strides=[1, 2, 2, 1],
                                                          padding='SAME'),
                                             b['be2'])),
                        keepprob)

    ce3 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d(ce2,
                                                          W['ce3'],
                                                          strides=[1, 2, 2, 1],
                                                          padding='SAME'),
                                             b['be3'])),
                        keepprob)

    # Decoder
    cd3 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(ce3,
                                                                    W['cd3'],
                                                                    tf.stack([tf.shape(X)[0], 7, 7, n2]),
                                                                    strides=[1, 2, 2, 1],
                                                                    padding='SAME'),
                                             b['bd3'])),
                        keepprob)

    cd2 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(cd3,
                                                                    W['cd2'],
                                                                    tf.stack([tf.shape(X)[0], 14, 14, n1]),
                                                                    strides=[1, 2, 2, 1],
                                                                    padding='SAME'),
                                             b['bd2'])),
                        keepprob)

    cd1 = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.nn.conv2d_transpose(cd2,
                                                                    W['cd1'],
                                                                    tf.stack([tf.shape(X)[0], 28, 28, 1]),
                                                                    strides=[1, 2, 2, 1],
                                                                    padding='SAME'),
                                             b['bd1'])),
                        keepprob)

    return {'input_r': image, 'ce1': ce1, 'ce2': ce2, 'ce3': ce3,
            'cd3': cd3, 'cd2': cd2, 'cd1': cd1,
            'layers': (image, ce1, ce2, ce3, cd3, cd2, cd1),
            'out': cd1}


x = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]])
y = tf.placeholder(tf.float32, [None, mnist.train.images.shape[1]])
keepprob = tf.placeholder(tf.float32)

pred = CNNAutoEncoder(x, weights, biases, keepprob)['out']

loss = tf.reduce_sum(tf.square(pred-tf.reshape(y, shape=[-1, 28, 28, 1])))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss)

init = tf.initialize_all_variables()

# Start Training
sess = tf.Session()
sess.run(init)
mean_img = np.zeros((784))
# Fit all training data

for epoch_i in range(n_epochs):
    for batch_i in range(mnist.train.num_examples // batch_size):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        trainbatch = np.array([img - mean_img for img in batch_xs])
        trainbatch_noisy = trainbatch + 0.3*np.random.randn(
            trainbatch.shape[0], 784)
        sess.run(optimizer, feed_dict={x: trainbatch_noisy
                                  , y: trainbatch, keepprob: 0.7})

    print("[%02d/%02d] cost: %.4f" % (epoch_i, n_epochs,
                                      sess.run(loss, feed_dict={x: trainbatch_noisy,
                                                                y: trainbatch,
                                                                keepprob: 1.})))
    if (epoch_i % 1) == 0:
        n_examples = 5
        test_xs, _ = mnist.test.next_batch(n_examples)
        test_xs_noisy = test_xs + 0.3*np.random.randn(
            test_xs.shape[0], 784)
        recon = sess.run(pred, feed_dict={x: test_xs_noisy, keepprob: 1.})
        fig, axs = plt.subplots(2, n_examples, figsize=(15, 4))
        for example_i in range(n_examples):
            axs[0][example_i].matshow(np.reshape(test_xs_noisy[example_i, :],
                                                 (28, 28)),
                                      cmap=plt.get_cmap('gray'))
            axs[1][example_i].matshow(np.reshape(np.reshape(recon[example_i, ...],
                                                            (784,)) + mean_img,
                                                 (28, 28)),
                                      cmap=plt.get_cmap('gray'))
        plt.show()
