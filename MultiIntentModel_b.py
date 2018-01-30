import sys
import numpy as np
import tensorflow as tf
from Data import Data

rnn_size = 10
batch_size = 3
word_length = 5
max_sentence_length = 3
lr = 0.02

inputs = [
    [
        [[1, 2, 3], [7, 8, 9]],
        [[2, 3, 4], [8, 9, 10]]
    ]
]
d = Data()
vocab_size = len(d.vocab) + 2
print(vocab_size)

data_x_y = d.run()
print(data_x_y)
sys.exit()

# inputs = np.array(inputs, dtype=np.float32)
train_graph = tf.Graph()
with train_graph.as_default():
    input_text = tf.placeholder(tf.int32, shape=(None, None), name="input")
    targets = tf.placeholder(tf.int32, shape=(None, None), name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    # Character RNN
    char_final_states = []
    embedding = tf.Variable(tf.random_uniform((vocab_size, rnn_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_text)
    print(embed.shape)
    reuse_ = False
    for n in range(max_sentence_length):
        with tf.variable_scope("chars", reuse=reuse_):
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="lstm")
            initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(lstm, embed, dtype=tf.float32)
            final_state = tf.identity(final_state, name="final_state_char_{}".format(n))
            print(final_state.shape)
            char_final_states.append(final_state)
        reuse_ = True

    # Atoms RNN
    char_final_states = tf.concat(char_final_states, axis=1)
    print(char_final_states.shape)
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="atoms")
    # initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(lstm, char_final_states, dtype=tf.float32)
    final_state_atoms = tf.identity(final_state, name="final_state_atoms")

    y = []
    fully_connected_logits = []
    relations_reuse = []
    items_reuse = []
    reuse_final_logits = False
    # A RNNs
    for item in d.items:
        if item not in items_reuse:
            reuse_ = False
            items_reuse.append(item)
        else:
            reuse_ = True
        with tf.variable_scope("items_a_c", reuse=reuse_):
            lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="lstm_a_{}".format(item))
            # initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
            outputs, final_state = tf.nn.dynamic_rnn(lstm, final_state_atoms, dtype=tf.float32)
            final_state_a = tf.identity(final_state, name="final_state_a_{}".format(item))
        # B RNN
        for relation in d.relations:
            if item not in relations_reuse:
                reuse_ = False
                relations_reuse.append(item)
            else:
                reuse_ = True
            with tf.variable_scope("relations_b", reuse=reuse_):
                lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="lstm_b_{}".format(item))
                # initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
                outputs, final_state = tf.nn.dynamic_rnn(lstm, final_state_a, dtype=tf.float32)
                final_state_b = tf.identity(final_state, name="final_state_b_{}".format(item))
            # C RNN
            for item_ in d.items:
                if item_ not in items_reuse:
                    reuse_ = False
                    items_reuse.append(item_)
                else:
                    reuse_ = True
                with tf.variable_scope("items_a_c", reuse=reuse_):
                    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, name="lstm_a_{}".format(item_))
                    # initial_state = lstm.zero_state(batch_size, dtype=tf.float32)
                    outputs, final_state = tf.nn.dynamic_rnn(lstm, final_state_b, dtype=tf.float32)
                    # y.append(tf.identity(final_state, name="final_state_a_{}".format(item_)))
                with tf.variable_scope("final_logits", reuse=reuse_final_logits):
                    reuse_final_logits = True
                    logits = tf.contrib.layers.fully_connected(outputs, 2, activation_fn=None)
                    logits = tf.identity(logits, name="final_logits")
                    fully_connected_logits.append(logits)
                    # probs = tf.nn.softmax(logits, name='probs_{}_{}_{}'.format(item, relation, item_))
    fully_connected_logits = tf.concat(fully_connected_logits, axis=1)
    print(fully_connected_logits.shape)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fully_connected_logits, labels=targets))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)


with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    # state = sess.run(initial_state, {"input_text": inputs[0]})
    state = sess.run(initial_state)
    x = inputs[0][0]
    y = inputs[0][1]
    feed = {
        input_text: x,
        targets: y,
        initial_state: state,
        learning_rate: lr
    }
    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
    # state = sess.run([final_state], feed)
