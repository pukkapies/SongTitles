import tensorflow as tf


def add_to_collection(**kwargs):
    for (k, v) in kwargs.items():
        if type(v) == list:
            for item in v:
                tf.add_to_collection(k, item)
        else:
            tf.add_to_collection(k, v)
