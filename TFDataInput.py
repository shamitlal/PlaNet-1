import tensorflow as tf
import os.path as path
import numpy as np
import random
from munch import Munch

import utils

import ipdb
st = ipdb.set_trace
#from tensorflow.examples.tutorials.mnist import input_data

'''
Note on the input system:
all input classes should inherit from Input and define a self.data method

the appropriate dataset (train/val/test) is selected as follows:
1. each mode (defined in models.py) has an associated data_name (train, val, or test)
2. each mode name maps to an index (0, 1, 2) respeectively

then one of the following will happen, depending on whether the graph is eager or not


Eager Mode:
3. the go method is called many times, once per iteration. an index is passed in
4. the go method usually calls something like self.prepare_data(index)
5. prepare_data calls the data method of the input class, passing in index
6. the data method selects the right data using a python conditional statement

- in eager mode, q_ph is ignored

it is difficult to have a unified input pipeline for both eager and graph mode because
1. we can't call/construct input tensors once per iteration in graph mode 
2. we can't use placeholders in eager mode

'''

class Input:
    '''
    the only function that every input class MUST define
    index specifies whether the data returned should be from the
    train, test, or validation set

    return a (dictionary of) tf tensors
    '''
    
    def data(self, index = None):
        raise NotImplementedError

class TFDataInput(Input):
    '''
    if you inherit from this class, you should define attributes
    self.train_data, self.test_data, self.val_data, which are objects
    with member functions .get_next(), which when called will return
    a *tf tensor* (or a dictionary/list of tensors) containing the data

    this supposed to be used with the tf.data API (in particular, the
    make_one_shot_iterator function)
    '''
    def __init__(self):
        pass
    def data(self, index = None):
        #use index to grab data from train, val, or test set
        assert index is not None
        return self.data_for_selector(tf.constant(index))

    def data_for_selector(self, selector):
        return [self.train_data.get_next,
                self.val_data.get_next,
                self.test_data.get_next][selector.numpy()]()
