import numpy as np


vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
         't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', "'", '(', ')', 'END']

assert len(set(vocab)) == len(vocab)

def get_vocab():
    return vocab

def get_vocab_size():
    return len(vocab)

def char_to_index(char):
    return vocab.index(char)


def index_to_char(index):
    return vocab[index]


def indices_to_arr(inx, dtype=np.int32):
    """:param inx: List of indices of vocab"""
    VOCAB_SIZE = get_vocab_size()
    seq_length = len(inx)
    arr = np.zeros(shape=(seq_length, VOCAB_SIZE), dtype=dtype)
    for char, index in enumerate(inx):
        arr[char, index] = 1
    return arr


def vec_to_char(vec):
    assert np.sum(vec) == 1 and (np.all(vec) in {0, 1})
    ind = np.where(vec == 1)[0]
    assert ind.size == 1
    ind = ind[0]
    return vocab[ind]


def arr_to_title(arr):
    assert len(arr.shape) == 2
    chars = []
    for row in arr:
        char = vec_to_char(row)
        chars.append(char)
    assert chars[-1] == 'END'
    title = ''.join(chars[:-1])
    return title


def process_title_string(title):
    """Puts capital letters at start of words"""
    chars = []
    previous_char = ''
    for i, char in enumerate(title):
        if i == 0 or previous_char in [' ', '(']:
            char = char.upper()
        chars.append(char)
        previous_char = char
    title = ''.join(chars)
    return title
