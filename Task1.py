import numpy as np
import tensorflow as tf
import matplotlib

# Parameters
CLASS_COUNT = 26
INPUT_COUNT = 16
TRAIN_VOLUME = 7000
TEST_VOLUME = 1000

learning_rate = 0.01
training_epochs = 250
batch_size = 100
display_step = 50

def read_data(filename):
    """ Read data from file.
        Return List of Turples (in this case pairs), last contains lists: input data and corresponding label"""
    data = []
    with open(filename) as f:
        content = f.readlines()
    for line in content:
        [class_str, vector_str] = line.split(',', 1)
        class_number = ord(class_str[0]) - ord('A')
        class_label = [0] * CLASS_COUNT
        class_label[class_number] = 1
        class_vector = [int(x) for x in vector_str.split(',')]
        data.append((class_vector, class_label))
    return data

n_input = INPUT_COUNT
n_hidden_1 = 100
n_hidden_2 = 100
n_classes = CLASS_COUNT

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

def mlp(_X, _weights, _biases):
    #Hidden layer with RELU activation
    #layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    #Hidden layer with RELU activation
    #layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2']))
    return tf.matmul(layer_2, weights['out']) + biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = mlp(x, weights, biases)

# Define loss and optimizer
# Softmax loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
# Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Mean square loss
#cost = tf.reduce_mean(tf.square(pred - y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Read data from file
data = read_data("letter-recognition.data")

train_data = data[0:TRAIN_VOLUME]
test_data = data[TRAIN_VOLUME:TRAIN_VOLUME+TEST_VOLUME]

train_vectors = []
train_labels = []
for (tv, tl) in train_data:
    train_vectors.append(tv)
    train_labels.append(tl)

test_vectors = []
test_labels = []
for (tv, tl) in train_data:
    test_vectors.append(tv)
    test_labels.append(tl)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data)/batch_size)
        k = 0
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, batch_ys = train_data[k:k+batch_size]
            batch_xs = train_vectors[k:k+batch_size]
            batch_ys = train_labels[k:k+batch_size]
            k += batch_size
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "Accuracy:", accuracy.eval({x: test_vectors, y: test_labels})
