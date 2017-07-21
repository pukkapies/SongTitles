import tensorflow as tf
import json
from random import shuffle
import shutil
import numpy as np
from utils.data_utils import load_data
from utils.setup import make_settings
from char_rnn import CharRNN



# For server
MODEL_FOLDER = '/home/kevin/pukkapies_github/SongTitles/saved_models/E128_L128_128_len/'
TRAINING_DATA_PATH = "/home/kevin/pukkapies_github/SongTitles/data/test_length/training/"
VALIDATION_DATA_PATH = "/home/kevin/pukkapies_github/SongTitles/data/test_length/validation/"

# For laptop
# MODEL_FOLDER = '/Users/kevinwebster/tensorflow/songtitles/saved_models/E128_L128_128_len/'
# TRAINING_DATA_PATH = "/Users/kevinwebster/tensorflow/songtitles/data/test_length/training/"
# VALIDATION_DATA_PATH = "/Users/kevinwebster/tensorflow/songtitles/data/test_length/validation/"

META_GRAPH = None  # "model-1750"


def main(MODEL_FOLDER=MODEL_FOLDER, TRAINING_DATA_PATH=TRAINING_DATA_PATH, VALIDATION_DATA_PATH=VALIDATION_DATA_PATH):
    if META_GRAPH:  # Restore an existing model
        print('\nrestoring model...')
        if MODEL_FOLDER[-1] != '/': MODEL_FOLDER += '/'
        if TRAINING_DATA_PATH[-1] != '/': TRAINING_DATA_PATH += '/'
        if VALIDATION_DATA_PATH[-1] != '/': VALIDATION_DATA_PATH += '/'

        with open(MODEL_FOLDER + 'settings.json') as json_file:
            settings = json.load(json_file)

        # Load training and (optional) validation data
        print('\nloading training and validation data...')
        # TODO: The data structure isn't set up for this type of network. We will only use the 'output' data
        # settings.update({'cuda_batch': 200, 'shuffle_before_batching': True})
        # data = load_data(settings)

        graph = tf.Graph()
        with graph.as_default():
            model = CharRNN(settings, graph, MODEL_FOLDER, meta_graph=META_GRAPH)
            print('... done.')

            generations = []
            for _ in range(100):
                generation = model.sample()
                print(generation)
                generations.append(generation)
            long_count = 0
            short_count = 0
            for generation in generations:
                if generation[:4] == 'good': long_count += 1
                elif generation[:3] == 'get': short_count += 1
                else: print('Generation not recognised: {}'.format(generation))
            print("Long count: ", long_count)
            print("Short count: ", short_count)

    else:  # Build a new model
        print("\nBuilding new model...")
        if MODEL_FOLDER[-1] != '/': MODEL_FOLDER += '/'
        if TRAINING_DATA_PATH[-1] != '/': TRAINING_DATA_PATH += '/'
        if VALIDATION_DATA_PATH[-1] != '/': VALIDATION_DATA_PATH += '/'
        cl_settings = {"training_data_path": TRAINING_DATA_PATH, 'validation_data_path': VALIDATION_DATA_PATH,
                       'output_folder': MODEL_FOLDER}
        settings = make_settings(cl_settings, MODEL_FOLDER)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # Load training and (optional) validation data
        print('\nloading training and validation data...')
        training_data, validation_data = load_data(settings)  # DatasetFeed object

        # #Test data
        # ((input_arr, input_len, input_mask), (target_arr, target_len, target_mask)) = training_data.next_batch()
        #
        # print(input_arr)
        # print(target_arr)
        # print("*********")
        # print(input_len)
        # print(target_len)
        # print("*********")
        # print(input_mask)
        # print(target_mask)
        #
        # for _ in range(5):
        #     ((input_arr, input_len, input_mask), (target_arr, target_len, target_mask)) = training_data.next_batch()
        #     print(input_arr.shape)
        #     print(target_arr.shape)

        # Build model
        print('\nbuilding model...')

        graph = tf.Graph()
        with graph.as_default():
            model = CharRNN(settings, graph, MODEL_FOLDER)
            print('... done.')
            model.train(settings, training_data=training_data, validation_data=validation_data)

if __name__=='__main__':
    main()
