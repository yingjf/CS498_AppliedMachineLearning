class ConvModel(nn.Module):
    # Your Code Here
    def __init__(self):
        super(ConvModel, self).__init__()
        weights = {
            'wc1': tf.get_variable('W0', shape=(3, 3, 1, 16), initializer=tf.contrib.layers.xavier_initializer()),
            'wc2': tf.get_variable('W1', shape=(3, 3, 16, 16), initializer=tf.contrib.layers.xavier_initializer()),
            'wd1': tf.get_variable('W3', shape=(7 * 7 * 16, 64), initializer=tf.contrib.layers.xavier_initializer()),
            'out': tf.get_variable('W6', shape=(64, n_classes), initializer=tf.contrib.layers.xavier_initializer()),
        }

    def forward(self, x):
        return conv_net(x, weights)

    def conv_net(x, weights):
        # here we call the conv2d function we had defined above and pass the input image x, weights wc1.
        conv1 = conv2d(x, weights['wc1'])
        # Apply ReLU
        conv1 = relu(conv1)
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
        conv1 = maxpool2d(conv1, k=2)

        # Convolution Layer
        # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
        conv2 = conv2d(conv1, weights['wc2'])
        # Apply ReLU
        conv2 = relu(conv2)
        # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']))
        # Apply ReLU
        fc1 = relu(fc1)

        # Output, class prediction
        # finally we multiply the fully connected layer with the weights and add a bias term.
        out = tf.add(tf.matmul(fc1, weights['out']))

        return out

    def conv2d(x, W, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        return tf.nn.relu(x)

    def relu(x):
        return tf.nn.relu(x)

    def maxpool2d(x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='S

        model = ConvModel().to(device)

        loss = nn.CrossEntropyLoss()
        optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.01)

        metrics = train(model, train_loader, test_loader, loss, optimizer, training_epochs)