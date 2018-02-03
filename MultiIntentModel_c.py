import sys
import numpy as np
import tensorflow as tf
from Data import Data

rnn_size = 128
# word_length = 5
# max_sentence_length = 60

batch_sizes = [100, 50, 10, 3, 1]
epochs = 100
lr = 0.002

d = Data(0)
vocab_size = len(d.vocab) + 2
print(vocab_size)

data_x, data_y = d.run()
questions_count = len(d.questions_all)
# print(data_x)
# print(data_y)
# print(data_x.shape)
# print(data_y.shape)
# sys.exit()

train_graph = tf.Graph()
with train_graph.as_default():
    input_text = tf.placeholder(tf.int32, shape=(None, None), name="input")
    targets = tf.placeholder(tf.int32, shape=(None, None), name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Character RNN
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="lstm")
    input_oh = tf.one_hot(input_text, vocab_size)
    outputs, _ = tf.nn.dynamic_rnn(lstm, input_oh, dtype=tf.float32)
    #
    outputs_b = tf.transpose(outputs, [1, 0, 2])
    # last = tf.gather(outputs_b, outputs_b[0])
    last = outputs_b[-1]
    #
    logits = tf.contrib.layers.fully_connected(last, rnn_size, activation_fn=None)
    logits = tf.contrib.layers.fully_connected(logits, questions_count * 1, activation_fn=None)
    logits = tf.identity(logits, name="final_logits")
    prediction = tf.nn.sigmoid(logits)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(targets, tf.float32), logits=logits)
    tf.summary.scalar('cost', tf.reduce_mean(cost))
    # train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
    train_op = optimizer.apply_gradients(capped_gradients)


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    for n in range(2):
        for bk in range(len(data_x)):
            x = data_x[bk]
            y = data_y[bk]
            # print("LEN", len(x))
            for batch_size in batch_sizes:
                if float(len(x)) / batch_size > 1.0:
                    break
            for n in range(epochs):
                for nn in range(0, len(x) - batch_size, batch_size):
                    feed = {
                        input_text: x[nn:nn + batch_size],
                        targets: y[nn:nn + batch_size],
                        learning_rate: lr
                    }
                    summary, pred, targ, train_loss, _ = sess.run([merged, prediction, targets, cost, train_op], feed)
                    train_writer.add_summary(summary, 0)
                    # print(x[0])
                    print("{} - {} - {}".format(batch_size, x[0].shape[0], train_loss.mean()))

    while 1:
        text_input = input(">")
        x = d.message_to_ints(text_input)
        feed = {
            input_text: [x],
        }
        pred = sess.run(prediction, feed)
        print(pred)
        for question in range(questions_count):
            if pred[0][question] > 0.5:
                print("{}, {}".format(d.questions_all[question], pred[0][question]))
