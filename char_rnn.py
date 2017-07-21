import logging
import time
import numpy as np
import tensorflow as tf
from utils.collections_handler import create_CharRNN_collections, unpack_CharRNN_handles
from utils.setup import create_json
from utils.vectorisation_utils import get_vocab
# Disable Tensorflow logging messages.
# logging.getLogger('tensorflow').setLevel(logging.WARNING)


class CharRNN(object):
    """Character RNN model."""

    RESTORE_KEY = 'to_restore/'

    def __init__(self, settings, graph, model_folder, meta_graph=None, scope='CharRNN'):
        # Shape [batch_size, seq_len, vocab_size]

        self.graph = graph
        self.meta_graph = meta_graph
        self.scope = scope
        assert model_folder[-1] == '/'
        self.model_folder = model_folder
        self.settings_path = self.model_folder + 'settings.json'
        self.collections = dict()
        self.vocab_size = settings['vocab_size']
        self.vocab = get_vocab()

        if meta_graph is not None:
            self.saver = tf.train.import_meta_graph(self.model_folder + meta_graph + ".meta")
            # print([var.name for var in tf.global_variables()])
        else:
            print("building new model...")
            CharRNN_settings = settings['CharRNN_settings']

            self.hidden_sizes = CharRNN_settings['architecture']
            self.embedding_size = CharRNN_settings['embedding_size']
            self.dropout = CharRNN_settings['dropout']
            with tf.variable_scope(self.scope) as graph_scope:
                # (batch_size, seq_length)
                self.input_ph = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ph')
                self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
                self.targets_one_hot = tf.one_hot(self.targets, self.vocab_size, on_value=1.0, off_value=0.0,
                                                  dtype=tf.float32)  # (batch_size, seq_len, vocab_size)

                self.loss_mask = tf.placeholder(dtype=tf.float32, shape=[None, None], name='loss_mask')
                self.batch_size = tf.shape(self.input_ph)[0]
                self.sequence_length = tf.shape(self.input_ph)[1]
                # Override object attributes with the correct graph, but reusing initialized variables:
                self._build_graph(CharRNN_settings)
                # print([var.name for var in tf.global_variables()])
                # graph_scope.reuse_variables()
            create_CharRNN_collections(self, CharRNN.RESTORE_KEY)

    def unpack_handles(self, session):
        unpack_CharRNN_handles(self, CharRNN.RESTORE_KEY, session)

    def save_model(self, outdir, settings, session):
        """Saves the model if a self.saver object exists"""
        try:
            outfile = outdir + 'model'
            self.saver.save(session, outfile, global_step=self.global_step.eval(session=session))
            create_json(self.settings_path, settings)
        except AttributeError:
            print("Failed to save model at step {}".format(self.global_step.eval(session=session)))

    def _build_graph(self, char_rnn_settings):
        cell_type = char_rnn_settings['cell_type']
        # batch_size = char_rnn_settings['batch_size']  # Used for training only

        if cell_type == 'rnn':
            cell_fn = tf.contrib.rnn.BasicRNNCell
        elif cell_type == 'lstm':
            cell_fn = tf.contrib.rnn.BasicLSTMCell
        elif cell_type == 'gru':
            cell_fn = tf.contrib.rnn.GRUCell
        else:
            raise AttributeError("Unrecognised recurrent cell type: {}".format(cell_type))

        cells = []
        for hidden_size in char_rnn_settings['architecture']:
            cell = cell_fn(hidden_size)
            cells.append(cell)

        self.cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)
        print("initial state: ", self.initial_state)

        with tf.name_scope('embedding_layer'):
            self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32)
            print("embedding shape: ", self.embedding.get_shape())
            print("input_ph: ", self.input_ph)
            inputs = tf.nn.embedding_lookup(self.embedding, self.input_ph)
            print("inputs: ", inputs)
            # if is_training and self.input_dropout > 0:
            #     inputs = tf.nn.dropout(inputs, 1 - self.input_dropout)

        outputs, final_state = tf.nn.dynamic_rnn(self.cell, inputs, initial_state=self.initial_state,
                                               dtype=tf.float32)
        self.final_state = final_state
        print('final state: ', final_state)
        print("outputs: ", outputs)  # (batch_size, seq_len, hidden_size)

        with tf.variable_scope("softmax_out"):
            softmax_w = tf.get_variable("softmax_w", [self.hidden_sizes[-1], self.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [self.vocab_size])
            flatten_outputs = tf.reshape(outputs, [-1, self.hidden_sizes[-1]], name="reshape_outputs")
            self.flat_logits = tf.matmul(flatten_outputs, softmax_w) + softmax_b  # (batch_size * seq_len, vocab_size)
            print("logits:", self.flat_logits.get_shape())
            self.flat_probs = tf.nn.softmax(self.flat_logits)
            print("flat probs: ", self.flat_probs)
            self.probs = tf.reshape(self.flat_probs, [self.batch_size, self.sequence_length, self.vocab_size])
            print("probs: ", self.probs)

        self.premasked_loss = self.cross_entropy(self.probs, self.targets_one_hot)
        print('loss: ', self.premasked_loss)
        print('loss_mask: ', self.loss_mask)
        self.masked_loss = self.premasked_loss * self.loss_mask
        print('masked loss: ', self.masked_loss)
        self.loss = tf.reduce_sum(self.masked_loss)

    @staticmethod
    def cross_entropy(obs, actual, offset=1e-7):
        """
        Binary cross-entropy, per training example.
        obs and actual are Tensors, shape (batch_size, seq_len, vector_size)
        :return: cross-entropy, averaged over vector_size: (batch_size, seq_len)
        """
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) + (1 - actual) * tf.log(1 - obs_), 2)

    def sample(self, max_seq_length=100):
        input = np.zeros(shape=[1, 1], dtype=np.float32)
        final_token = False
        chars = []

        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, self.model_folder + self.meta_graph)
            self.unpack_handles(sess)
            # print([var.name for var in tf.global_variables()])
            init_state = sess.run(self.initial_state, feed_dict={self.input_ph: input})

            while final_token is False:
                output, state = sess.run([self.probs, self.final_state],
                                             feed_dict={self.input_ph: input, self.initial_state: init_state})
                output = np.squeeze(output)
                # print(output)
                char = np.random.choice(self.vocab, p=output)
                if char == 'END':
                    return ''.join(chars)
                else:
                    chars.append(char)
                    index = self.vocab.index(char)
                    input = np.ones(shape=[1, 1], dtype=np.float32) * index
                    init_state = state
                    if len(chars) >= max_seq_length:
                        return ''.join(chars)

    def train(self, settings, training_data, validation_data):
        # if is_training and self.dropout > 0:
        #     cells = [tf.contrib.rnn.DropoutWrapper(
        #         cell,
        #         output_keep_prob=1.0 - self.dropout)
        #              for cell in cells]
        self.global_step = tf.get_variable("global_step", shape=[], dtype=tf.int32,
                                           initializer=tf.zeros_initializer(), trainable=False)
        optimiser = tf.train.AdamOptimizer(learning_rate=settings['learning_rate'])
        tvars = tf.trainable_variables()
        grads_and_vars = optimiser.compute_gradients(self.loss, tvars)

        clipped = [(tf.clip_by_value(grad, -5, 5), tvar) if grad is not None else (grad, tvar)  # gradient clipping
                   for grad, tvar in grads_and_vars]
        self.train_op = optimiser.apply_gradients(clipped, global_step=self.global_step, name="minimize_cost")
        self.saver = tf.train.Saver(max_to_keep=50)

        patience = 0
        best_valid_loss = np.inf
        best_train_loss = np.inf

        results = dict()

        with tf.Session(graph=self.graph) as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            max_epochs = settings['max_epoch']
            epoch = 0
            results['epoch'] = int(epoch)
            train_losses = []
            valid_losses = []

            while True:
                ((input_arr, input_len, input_mask), (target_arr, target_len, target_mask)) = training_data.next_batch()

                # For validation
                ((v_input_arr, v_input_len, v_input_mask),
                 (v_target_arr, v_target_len, v_target_mask)) = validation_data.next_batch()
                validation_loss = sess.run([self.loss], feed_dict={self.input_ph: v_input_arr,
                                                                   self.targets: v_target_arr,
                                                                   self.loss_mask: v_input_mask})

                # premasked_loss, masked_loss = sess.run([self.premasked_loss, self.masked_loss],
                #                                        feed_dict={self.input_ph: input_arr,
                #                                                   self.targets: target_arr,
                #                                                   self.loss_mask: input_mask})
                #
                # print(premasked_loss)
                # print(masked_loss)

                train_loss, _ = sess.run([self.loss, self.train_op], feed_dict={self.input_ph: input_arr,
                                                                                self.targets: target_arr,
                                                                                self.loss_mask: input_mask})
                train_losses.append(train_loss)

                # print("Loss shape: ", train_loss.shape)
                # print(input_arr)
                # print(input_mask)

                valid_losses.append(validation_loss)

                step = sess.run(self.global_step)
                # print("step: {}".format(step))

                if training_data.epochs_completed > epoch:  # End of an epoch
                    average_train_loss = np.mean(train_losses)
                    epoch = training_data.epochs_completed
                    results['epoch'] = int(epoch)
                    print("    Average training loss for this epoch: {}".format(average_train_loss))
                    results['train_loss'] = float(average_train_loss)
                    train_losses = []

                    # Do validation
                    average_valid_loss = np.mean(valid_losses)
                    valid_losses = []
                    print("    Average validation loss for this epoch: {}".format(average_valid_loss))
                    if average_valid_loss < best_valid_loss:
                        results['best_validation_cost'] = float(average_valid_loss)
                        # Update best loss, save model, set patience to 0.
                        best_valid_loss = average_valid_loss
                        if step > 500000:  # Save the model at every time at this point
                            self.save_model(self.model_folder, settings, sess)
                            create_json(self.model_folder + 'results.json', results)
                            print('Model saved in %s' % self.model_folder)
                        patience = 0
                    else:
                        # Increment patience and check if training is to be stopped.
                        patience += 1
                        print("Patience = {}".format(patience))

                        if patience == settings['max_patience']:
                            print('patience threshold of %d reached, exiting...'
                                  % (settings['max_patience']))
                            break

                    if (epoch -1 ) % 2 == 0:  # Save the model every 10 epochs
                        if average_train_loss < best_train_loss:
                            self.save_model(self.model_folder, settings, sess)
                            best_train_loss = average_train_loss
                            create_json(self.model_folder + 'results.json', results)
                            print('Model saved in %s' % self.model_folder)
                            # patience = 0
                        else:
                            pass
                            # patience += 1
                            # print('patience = {}'.format(patience))
                            # if patience == settings['max_patience']:
                            #     print('patience threshold of %d reached, exiting...'
                            #           % (settings['max_patience']))
                            #     break
                if epoch >= max_epochs: break
