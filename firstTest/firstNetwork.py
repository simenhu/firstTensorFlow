import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 100
n_nodes_hl2 = 100
n_nodes_hl3 = 100

n_classes = 10
batch_size = 100

x = tf.placeholder("float", [None, 784], name = "input-data")
y = tf.placeholder("float", name="data-label")
pkeep = tf.placeholder("float")

def neural_network_model(data, pkeep):

    hiddel_1_layer = {"weights": tf.Variable(tf.random_normal([784, n_nodes_hl1]), name="h1-weights"),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}

    tf.summary.histogram("h1-weights", hiddel_1_layer["weights"])

    hiddel_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]), name="h2-weights"),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    tf.summary.histogram("h2-weights", hiddel_2_layer["weights"])

    hiddel_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]), name="h3-weights"),
                      "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    tf.summary.histogram("h3-weights", hiddel_3_layer["weights"])


    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    "biases": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hiddel_1_layer["weights"]), hiddel_1_layer["biases"])
    l1 = tf.nn.dropout(tf.nn.relu(l1), pkeep)


    l2 = tf.add(tf.matmul(l1, hiddel_2_layer["weights"]), hiddel_2_layer["biases"])
    l2 = tf.nn.dropout(tf.nn.relu(l2), pkeep)

    l3 = tf.add(tf.matmul(l2, hiddel_3_layer["weights"]), hiddel_3_layer["biases"])
    l3 = tf.nn.dropout(tf.nn.relu(l3), pkeep)


    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

    return output

def train_neural_network(x):
    train_prediction = neural_network_model(x, pkeep)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    merged = tf.summary.merge_all()
    train_writer = tf.train.SummaryWriter("/")

    hm_epochs = 30

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        train_writer = tf.train.SummaryWriter("/", sess.graph)

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c, summary = sess.run([optimizer, cost, merged], feed_dict={x: epoch_x, y: epoch_y, pkeep: 1.0})
                epoch_loss+= c
                if _ == 1:
                    train_writer.add_summary(summary, epoch)
            print("epoch", epoch, "completed out of", hm_epochs,"loss:", epoch_loss, "optimizer output:", _)

        correct = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, "float"))
        print("Accuracy:", accuracy.eval({x:mnist.test.images, y:mnist.test.labels, pkeep: 1.0}))

train_neural_network(x)