# slice text in equal sections (with padding)
# run a RNN with shared parameters on each section
# take the resulting states in one section (with padding)
# run a RNN with shared parameters on the section
# take the result from that RNN

import numpy as np
import tensorflow as tf
from Data import Data
# from tensorflow.contrib import seq2seq
# words_in_dataset = tf.placeholder(tf.float32, [time_steps, batch_size, num_features])
# lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    # vocab = list(set(' '.join(text).split(' ')))
    vocab_to_int = dict((v, k + 1) for (k, v) in enumerate(vocab))
    int_to_vocab = dict((k + 1, v) for (k, v) in enumerate(vocab))
    return (vocab_to_int, int_to_vocab)


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    input_ = tf.placeholder(tf.int32, shape=(None, None), name="input")
    targets = tf.placeholder(tf.int32, shape=(None, None), name="targets")
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    return input_, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * 2)
    initialize_state = tf.identity(cell.zero_state(batch_size, tf.float32), name="initial_state")
    return cell, initialize_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, input_data)
    return embed


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")
    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    inputs = get_embed(input_data, vocab_size, rnn_size)
    rnn_outputs, final_state = build_rnn(cell, inputs)
    logits = tf.contrib.layers.fully_connected(rnn_outputs, vocab_size, activation_fn=None)
    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    number_of_batches = int(len(int_text) / (batch_size * seq_length))
    np_batchs = np.ndarray(shape=(number_of_batches, 2, batch_size, seq_length), dtype=np.int32)
    for n in range(number_of_batches):
        for nn in range(2):
            for nnn in range(batch_size):
                start = (n * (batch_size + seq_length + 1)) + (nnn * seq_length)
                np_batchs[n, nn, nnn] = int_text[start + nn:start + seq_length + nn]
    return np_batchs

# Number of Epochs
num_epochs = 130
# Batch Size
batch_size = 256
# RNN Size
rnn_size = 128
# Sequence Length
seq_length = 50
# Learning Rate
learning_rate = 0.02
# Show stats for every n number of batches
show_every_n_batches = 25



if 0:
    vocab_to_int, int_to_vocab = create_lookup_tables(vocab)

    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size)
        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')
        # Loss function
        # cost = seq2seq.sequence_loss(
        #     logits,
        #     targets,
        #     tf.ones([input_data_shape[0], input_data_shape[1]]))
        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        opt_op = opt.minimize(cost, var_list=<list of variables>)
        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        train_op = optimizer.apply_gradients(capped_gradients)

    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})
            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))
        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')
