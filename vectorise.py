from utils.vectorisation_utils import *
import os
import numpy as np


DATA_FOLDER = '/Users/kevinwebster/tensorflow/songtitles/data/second/'


if __name__ == '__main__':
    if DATA_FOLDER[-1] != '/': DATA_FOLDER += '/'
    assert os.path.exists(DATA_FOLDER), "Data folder does not exist!"

    VEC_OUTPUT_FOLDER = DATA_FOLDER + 'training/'
    if not os.path.exists(VEC_OUTPUT_FOLDER):
        os.makedirs(VEC_OUTPUT_FOLDER)

    vec_counter = 0

    for filename in os.listdir(DATA_FOLDER):
        if os.path.isdir(os.path.join(DATA_FOLDER, filename)) or (filename == '.DS_Store'):
            continue

        VOCAB_SIZE = get_vocab_size()

        with open(DATA_FOLDER + filename, 'r') as data_file:
            lines = data_file.read().split('\n')  # list of titles as strings

        for i, title in enumerate(lines):
            if len(title) == 0:
                print("Empty title found at line {}".format(i))
                continue
            inx = []
            try:
                for char in title:
                    if char == '\xe2':
                        inx.append(char_to_index("'"))
                    elif char == '\x80':
                        continue
                    elif char == '\x99':
                        continue
                    else:
                        inx.append(char_to_index(char))
                inx.append(char_to_index('END'))

                seq_length = len(inx)
                np_inx = np.array(inx, dtype=np.int32)

                np.savetxt(VEC_OUTPUT_FOLDER + '{}.vec'.format(vec_counter), np_inx, fmt='%d')
                vec_counter += 1
            except:
                # print([char for char in title])
                print("Vectorisation failed in filename {}, title number {}".format(filename, i + 1))


