import pickle
from sklearn.model_selection import train_test_split

training_file = "data/train.p"
testing_file = "data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, X_validation, y_train, y_validation = train_test_split(train['features'], train['labels'], test_size=0.2)
X_test, y_test = test['features'], test['labels']

import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

EPOCHS = 30
BATCH_SIZE = 50
DROPOUT = 0.75

def weight_variable(shape):
    initial = tf.truncated_normal(shape, mean=0, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
def conv_net(x, dropout):
    # Convolution Layer 1. 32x32x3 -> 32x32x6
    conv1_W = weight_variable([5, 5, 3, 6])
    conv1_b = bias_variable([6])
    conv1 = conv2d(x, conv1_W, conv1_b)
    
    # Pooling Layer 1. 32x32x6 -> 16x16x6
    conv1 = maxpool2d(conv1)
    # conv1 = tf.nn.dropout(conv1, dropout)
    
    # Convolution Layer 2. 16x16x6 -> 16x16x16
    conv2_W = weight_variable([5, 5, 6, 16])
    conv2_b = bias_variable([16])
    conv2 = conv2d(conv1, conv2_W, conv2_b)
    
    # Pooling Layer 2. 16x16x16 -> 8x8x16
    conv2 = maxpool2d(conv2)
    # conv2 = tf.nn.dropout(conv2, dropout)
    
    # Fully Connected Layer 1. 4x4x32 -> 120
    fc1_W = weight_variable([8*8*16, 120])
    fc1_b = bias_variable([120])
    fc1 = tf.reshape(conv2, [-1, 8*8*16])
    fc1 = tf.nn.relu(tf.matmul(fc1, fc1_W) + fc1_b)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Fully Connected Layer 2. 120 -> 84
    fc2_W = weight_variable([120, 84])
    fc2_b = bias_variable([84])
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_W) + fc2_b)
    
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)
    
    # Fully Connected Layer 3. 84 -> 43
    fc3_W = weight_variable([84, 43])
    fc3_b = bias_variable([43])
    fc3 = tf.matmul(fc2, fc3_W) + fc3_b
    return fc3

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)
keep_prob = tf.placeholder(tf.float32)

logits = conv_net(x, keep_prob)
loss_operation = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y))
optimizer = tf.train.AdamOptimizer()
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data, dropout):
    num_examples = len(X_data)
    total_accuracy, total_loss = 0, 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy =  sess.run([loss_operation, accuracy_operation], feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        total_accuracy += (accuracy * batch_x.shape[0])
        total_loss     += (loss * batch_x.shape[0])
    return total_loss / num_examples, total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            loss = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: DROPOUT})
            
        validation_loss, validation_accuracy = evaluate(X_validation, y_validation, 1.)
        print("EPOCH {} ...".format(i+1))
        print("Validation Loss     = {:.3f}".format(validation_loss))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    test_loss, test_accuracy = evaluate(X_test, y_test, 1.)
    print("Test Loss     = {:.3f}".format(test_loss))
    print("Test Accuracy = {:.3f}".format(test_accuracy))