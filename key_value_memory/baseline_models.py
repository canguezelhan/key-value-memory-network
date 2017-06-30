from __future__ import absolute_import
from __future__ import print_function

import logging
import time
from itertools import chain
from collections import Counter

import numpy as np
import tensorflow as tf
from six.moves import range, reduce
from sklearn import metrics

from data_utils import load_data


timestamp = str(int(time.time()))

tf.flags.DEFINE_string("baseline", "corpus", "corpus or context")
tf.flags.DEFINE_string("data", "cbt", "The data set (i.e., CBT, SQuAD).")
tf.flags.DEFINE_float("training_percentage", 1.0, "The percentage of the training data set to use.")
tf.flags.DEFINE_float("testing_percentage", 1.0, "The percentage of the testing data set to use.")
tf.flags.DEFINE_string("memory_representation", "sentence", "Memory representation of memory (i.e., sentence, window).")
tf.flags.DEFINE_integer("window_size", 5, "Size of a memory slot in a window-level memory.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug_mode", False, "Activate debug mode, which prints some info about the model.")

FLAGS = tf.flags.FLAGS

# Verify the validity of given parameters
assert FLAGS.data in ['cbt', 'squad', 'cnn'], 'Wrong input for data: {} given, cbt, squad or cnn expected.'.format(FLAGS.data)
assert 0 < FLAGS.training_percentage <= 1, 'Wrong input for training_percentage: {} given, a value in (0 ; 1] expected.'.format(FLAGS.training_percentage)
assert 0 < FLAGS.testing_percentage <= 1, 'Wrong input for testing_percentage: {} given, a value in (0 ; 1] expected.'.format(FLAGS.testing_percentage)

data_set = {'cbt': 'CBTest', 'squad': 'SQuAD', 'cnn': 'cnn'}
path = "results0/{}/xxx/".format(data_set[FLAGS.data])
log_dir = path.replace('xxx', 'logs', 1)

# Name of the logs file
log_file = "{}log_{}.txt".format(log_dir, timestamp)
# Configuration of the logging system
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

# Name of the parameters file
param_output_file = "{}params_{}.csv".format(log_dir, timestamp)

FLAGS._parse_flags()
logger.info("Parameters:")
with open(param_output_file, 'w') as f:
    for attr, value in sorted(FLAGS.__flags.items()):
        line = "{}={}".format(attr.upper(), value)
        f.write(line + '\n')
        logger.info(line)
    logger.info("")

logger.info("Started Program")

# Data
train, test = load_data(FLAGS.data,
                        FLAGS.training_percentage, FLAGS.testing_percentage,
                        FLAGS.memory_representation, FLAGS.window_size)
data = train + test
n_test = len(test) / 4

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + c + a) for s, q, c, a in data)))  # if FLAGS.data == 'cbt' else sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
if None in vocab:
    vocab.remove(None)

token_counter = Counter(list(chain.from_iterable(list(list(chain.from_iterable(s)) + q + c + a) for s, q, c, a in data)))

pred = []
if FLAGS.baseline == "corpus":
    for elem in test:
        tmp = []
        for w in elem[2]:
            tmp.append(token_counter[w])
        pred.append(elem[2][np.argmax(tmp)])
else:
    for elem in test:
        context_counter = Counter(list(list(chain.from_iterable(elem[0])) + elem[1] + elem[2] + elem[3]))
        tmp = []
        for w in elem[2]:
            tmp.append(context_counter[w])
        pred.append(elem[2][np.argmax(tmp)])

labels = [elem[3][0] for elem in test]
if FLAGS.data == 'cbt':
    word_types = {0: 'NE', 1: 'CN', 2: 'V', 3: 'P'}
    for i in range(4):
        acc = metrics.accuracy_score(labels[i*n_test:(i+1)*n_test], pred[i*n_test:(i+1)*n_test])
        logger.info('Testing Accuracy ({0}): {1:.3f}'.format(word_types[i], acc))
else:
    acc = metrics.accuracy_score(labels, pred)
    logger.info('Testing Accuracy: {0:.3f}'.format(acc))
