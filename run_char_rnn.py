import tensorflow as tf
import json
from random import shuffle
import shutil
import numpy as np
from utils.data_utils import load_data
from utils.setup import make_settings
from char_rnn import CharRNN



# For server
# MODEL_FOLDER = '/home/kevin/Jukedeck/composition/composition/training/vae/saved_models/L2_z2_h256_AFprior_3IAF_majmin/'
# DATASET_PATH = "/home/kevin/Jukedeck/composition/polyphonic_vectors/alkis_non_triadic/both/"

# For laptop
REPO_FOLDER =  '/Users/kevinwebster/jukedeck/composition'
# MODEL_FOLDER = REPO_FOLDER + '/composition/training/vae/saved_models/alkis_non_triadic_major/8 - condition_chords/test'
MODEL_FOLDER = '/Users/kevinwebster/tensorflow/songtitles/saved_models/test/'
TRAINING_DATA_PATH = "/Users/kevinwebster/tensorflow/songtitles/data/test/training/"
VALIDATION_DATA_PATH = "/Users/kevinwebster/tensorflow/songtitles/data/test/validation/"

META_GRAPH = "model-1848"


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
            # posterior_data is a list of posterior_epss for each category. prior_data is list [prior_eps, prior_z]

            print(model.sample())
            print(model.sample())
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
