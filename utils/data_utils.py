import numpy as np
from random import shuffle
import os
from utils.vectorisation_utils import get_vocab_size


def load_data(settings):
    """Returns training and validation DatasetFeed objects"""
    training_data_path = settings['training_data_path']
    validation_data_path = settings['validation_data_path']

    training_data_list = load_data_list(training_data_path)
    validation_data_list = load_data_list(validation_data_path)

    training_dataset = DatasetFeed(training_data_list, settings['cuda_batch'], shuffle_after_every_epoch=settings['shuffle_after_every_epoch'])

    settings.update({'vocab_size': training_dataset.vocab_size})

    validation_dataset = DatasetFeed(validation_data_list, len(validation_data_list),
                                     shuffle_after_every_epoch=False, print_epoch=False)
    return training_dataset, validation_dataset


class DatasetFeed(object):

    def __init__(self, loaded_list, minibatch_size, shuffle_after_every_epoch=True, print_epoch=True):
        """
        Initialiser
        :param loaded_list: List of numpy arrays shape (seq_len, ) for training
        """
        self.check_loaded_list(loaded_list)
        self.print_epoch = print_epoch
        self.vocab_size = get_vocab_size()
        print("Vocab size: ", self.vocab_size)
        self.shuffle_after_every_epoch = shuffle_after_every_epoch
        self.data = self.create_inputs_and_targets(loaded_list)  # Tuple (inputs, targets)
        self.num_data_points = len(self.data[0])
        print("Number of data points loaded: ", self.num_data_points)
        self.minibatch_size = minibatch_size
        assert self.minibatch_size <= self.num_data_points, "Data minibatch must be less than the number of data points"
        self.epochs_completed = 0
        self.current_dataset_index = 0

    @staticmethod
    def check_loaded_list(loaded_list):
        assert type(loaded_list) == list
        for item in loaded_list:
            assert type(item) == np.ndarray

    def create_inputs_and_targets(self, loaded_list):
        """Feeds a zero vector at the beginning of the input."""
        inputs = []
        targets = loaded_list
        for piece in loaded_list:
            shape = piece.shape
            input = np.hstack((np.array([0], dtype=np.int32), piece[:-1]))
            assert input.shape == shape
            inputs.append(input)
        assert len(inputs) == len(targets)
        return inputs, targets

    def next_batch_list(self):
        """
        Returns the next minibatch
        :return: Tuple of 2 np.ndarrays, shape (batch_size, vocab_size), for inputs and targets
        """
        current_index = self.current_dataset_index
        next_index = self.current_dataset_index + self.minibatch_size
        if next_index < self.num_data_points:  # next_index still points within the range of data points
            self.current_dataset_index = next_index
            return (np.asarray(self.data[0][current_index: next_index]),
                    np.asarray(self.data[1][current_index: next_index]))
        else:
            self.current_dataset_index = next_index % self.num_data_points
            self.epochs_completed += 1
            if self.print_epoch:
                print("Completed {} epochs".format(self.epochs_completed))
            first_input_sub_batch = self.data[0][current_index:]  # The remainder of the current set of data points
            first_target_sub_batch = self.data[1][current_index:]  # The remainder of the current set of data points
            if self.shuffle_after_every_epoch:
                combined = list(zip(self.data[0], self.data[1]))
                shuffle(combined)
                self.data[0][:], self.data[1][:] = zip(*combined)
            return (np.asarray(first_input_sub_batch + self.data[0][:self.current_dataset_index]),
                    np.asarray(first_target_sub_batch + self.data[1][:self.current_dataset_index]))

    def pack_minibatch_list_into_array(self, minibatch_list):
        seq_lengths = [datapt.shape[0] for datapt in minibatch_list]
        lengths = np.array(seq_lengths, dtype=np.int32)

        max_length = max(seq_lengths)
        mask = np.zeros(shape=(self.minibatch_size, max_length), dtype=np.float32)
        for row, length in enumerate(seq_lengths):
            mask[row, :length] = 1.0

        minibatch_array = np.zeros(shape=(self.minibatch_size, max_length), dtype=np.int32)

        for piece, array in enumerate(minibatch_list):
            minibatch_array[piece, :lengths[piece]] = minibatch_list[piece]

        return minibatch_array, lengths, mask

    def next_batch(self):
        next_input_batch_list, next_target_batch_list = self.next_batch_list()
        return (self.pack_minibatch_list_into_array(next_input_batch_list),
                self.pack_minibatch_list_into_array(next_target_batch_list))


def load_data_list(path):
    print("Loading data from {}".format(path))
    data_list = []
    for filename in os.listdir(path):
        if filename[-4:] != '.vec':
            continue
        piece = np.loadtxt(path + filename, dtype=np.int32)
        data_list.append(piece)
    print("Loaded {} vectorised pieces".format(len(data_list)))
    return data_list


def load_data_txt(path):
    title_strings = []
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)) or (filename == '.DS_Store'):
            continue

        with open(path + filename, 'r') as data_file:
            lines = data_file.read().split('\n')  # list of titles as strings
            title_strings.extend(lines)
    return title_strings


def check_data_for_quote(title, loaded_list):
    for data_title in loaded_list:
        if data_title == title:
            return True
    return False
