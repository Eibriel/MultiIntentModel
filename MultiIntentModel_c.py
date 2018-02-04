import sys
import random
import numpy as np
import tensorflow as tf
from Data import Data

rnn_size = 128
# word_length = 5
# max_sentence_length = 60

batch_sizes = [100, 50, 10, 3, 1]
epochs = 3
lr = 0.002
use_embed = True
load_checkpoint = False

d = Data(0)
vocab_size = len(d.vocab) + len(d.special.keys())
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
    if use_embed:
        embedding = tf.Variable(tf.random_uniform((vocab_size, rnn_size), -1, 1))
        input_rnn = tf.nn.embedding_lookup(embedding, input_text)
    else:
        input_rnn = tf.one_hot(input_text, vocab_size)
    outputs, _ = tf.nn.dynamic_rnn(lstm, input_rnn, dtype=tf.float32)
    #
    outputs_b = tf.transpose(outputs, [1, 0, 2])
    last = outputs_b[-1]
    #
    logits = tf.contrib.layers.fully_connected(last, rnn_size, activation_fn=None)
    logits = tf.contrib.layers.fully_connected(logits, questions_count, activation_fn=None)
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
    saver = tf.train.Saver()
    if load_checkpoint:
        saver.restore(sess, tf.train.latest_checkpoint("./02"))
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    step = 0
    # Ordered sequence
    sequence = list(range(len(data_x)))
    for n in range(epochs):
        for nn in sequence:
            feed = {
                input_text: data_x[nn],
                targets: data_y[nn],
                learning_rate: lr
            }
            summary, pred, targ, train_loss, _ = sess.run([merged, prediction, targets, cost, train_op], feed)
            step += 1
            train_writer.add_summary(summary, step)
            print("{} - {} - {}".format(data_x[nn].shape[0], data_x[nn].shape[1], train_loss.mean()))
        random.shuffle(sequence)
    saver.save(sess, "./checkpoints/my-model", global_step=step)

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
