from __future__ import absolute_import
from __future__ import print_function

import logging
import time
from itertools import chain
from collections import Counter

import numpy as np
import tables
import tensorflow as tf
from six.moves import range, reduce
from sklearn import metrics

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pylab as plt

from data_utils import load_data, vectorize_data, tsne_viz
from memn2n_kv import get_glove, position_encoding, identity_encoding, MemN2N_KV


timestamp = str(int(time.time()))

tf.flags.DEFINE_boolean("discard_rare_words", False, "Discard all the words that have a term frequency lower than a threshold.")
tf.flags.DEFINE_integer("term_freq_thr", 100, "Term frequency threshold.")
tf.flags.DEFINE_boolean("l2", False, "Use L2 regularization.")
tf.flags.DEFINE_float("l2_lambda", 0.1, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_boolean('add_gradient_noise', False, "Add gradient noise for training the model.")
tf.flags.DEFINE_boolean('anneal_noise', False, "Anneal gradient noise at each training step.")
tf.flags.DEFINE_boolean("gradient_clipping", False, "Use gradient clipping by norm.")
tf.flags.DEFINE_float("max_grad_norm", 100.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 3, "Number of hops in the Memory Network.")
tf.flags.DEFINE_integer("epochs", 200, "Maximum number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 100, "Embedding size for embedding matrices.")
tf.flags.DEFINE_boolean("glove", False, "Use the pre-trained GloVe vectors.")
tf.flags.DEFINE_boolean("dropout", False, "Use the dropout regularization.")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep probability for dropout.")
tf.flags.DEFINE_integer("memory_size", -1, "Maximum size of memory. -1 to use all of them.")
tf.flags.DEFINE_string("memory_representation", "sentence", "Memory representation of memory (i.e., sentence, window).")
tf.flags.DEFINE_integer("window_size", 5, "Size of a memory slot in a window-level memory.")
tf.flags.DEFINE_string("data", "cbt", "The data set (i.e., CBT, SQuAD).")
tf.flags.DEFINE_float("training_percentage", 1.0, "The percentage of the training data set to use.")
tf.flags.DEFINE_float("testing_percentage", 1.0, "The percentage of the testing data set to use.")
tf.flags.DEFINE_string("reader", "bow", "Reader for the model (bow, simple_gru)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug_mode", False, "Activate debug mode, which prints some info about the model.")

FLAGS = tf.flags.FLAGS

# Verify the validity of given parameters
assert FLAGS.data in ['cbt', 'squad', 'cnn'], 'Wrong input for data: {} given, cbt, squad or cnn expected.'.format(FLAGS.data)
assert 0 < FLAGS.training_percentage <= 1, 'Wrong input for training_percentage: {} given, a value in (0 ; 1] expected.'.format(FLAGS.training_percentage)
assert 0 < FLAGS.testing_percentage <= 1, 'Wrong input for testing_percentage: {} given, a value in (0 ; 1] expected.'.format(FLAGS.testing_percentage)
# if FLAGS.data.lower() == 'cbt':
#     assert FLAGS.word_type in ['NE', 'CN', 'V', 'P'], 'Wrong input for word_type: {} given, NE, CN, P or V expected.'.format(FLAGS.word_type)
assert FLAGS.memory_representation in ['sentence', 'window'], 'Wrong input for memory_representation: {} given, sentence or window expected.'.format(FLAGS.memory_representation)
if FLAGS.memory_representation == 'window':
    assert FLAGS.window_size % 2 == 1, 'Wrong input for window_size: {} given, odd integer expected.'.format(FLAGS.window_size)
if FLAGS.dropout:
    assert 0 < FLAGS.keep_prob <= 1, 'Wrong input for keep_prob: {} given, a value in (0 ; 1] expected.'.format(FLAGS.keep_prob)
if FLAGS.glove:
    assert FLAGS.embedding_size in [50, 100, 200, 300], 'Wrong input for embedding_size: {} given, 50, 100, 200 or 300 expected.'.format(FLAGS.embedding_size)

data_set = {'cbt': 'CBTest', 'squad': 'SQuAD', 'cnn': 'cnn'}
path = "results0/{}/xxx/{}/".format(data_set[FLAGS.data], FLAGS.memory_representation)
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

vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + c + a) for s, q, c, a in data)))  # if FLAGS.data == 'cbt' else sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
if None in vocab:
    vocab.remove(None)

if FLAGS.discard_rare_words:
    token_counter = Counter(list(chain.from_iterable(list(list(chain.from_iterable(s)) + q + c + a) for s, q, c, a in data)))
    if None in token_counter:
        del token_counter[None]
    logger.info("Minimum term frequency {}".format(np.amin(token_counter.values())))
    logger.info("Maximum term frequency {}".format(np.amax(token_counter.values())))
    logger.info("Average term frequency {0:.1f}".format(np.mean(token_counter.values())))
    logger.info("Median term frequency {0:.1f}".format(np.median(token_counter.values())))
    answer_candidates = sorted(reduce(lambda x, y: x | y, (set(c + a) for s, q, c, a in data)))
    for key, count in token_counter.most_common():
        if count >= FLAGS.term_freq_thr or key in answer_candidates:
            del token_counter[key]
    logger.info("{} / {} tokens have a term frequency less than {} (answer candidates are excluded).".format(len(token_counter), len(vocab), FLAGS.term_freq_thr))

    train, test = load_data(FLAGS.data,
                            FLAGS.training_percentage, FLAGS.testing_percentage,
                            FLAGS.memory_representation, FLAGS.window_size,
                            token_counter)
    data = train + test

    vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + c + a) for s, q, c, a in data)))  # if FLAGS.data == 'cbt' else sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + q + a) for s, q, a in data)))
    if None in vocab:
        vocab.remove(None)

# The dictionary of words mapping to their respective index
word2idx = dict((c, i + 1) for i, c in enumerate(vocab))
# The inverse dictionary
idx2word = dict(zip(word2idx.values(), word2idx.keys()))

count = 0
indices, updates = get_glove(FLAGS.embedding_size, word2idx, logger)
for s, q, c, a in data:
    if not word2idx[a[0]] in indices:
        count += 1
logger.info("{} / {} answers do not have a vector representation in GloVe.".format(count, len(data)))

# Maximum number of memory slots for a query
max_story_size = 0
# Average value of the number of memory slots per query
mean_story_size = 0.0
# Maximum number of words in a memory slot
sentence_size = 0
# Maximum number of words in a query
query_size = 0
# Maximum number of answer candidates for a query
max_candidates_size = 0
# Average number of answer candidates per query
mean_candidates_size = 0.0

for i, element in enumerate(data):
    # Context of the question-answer pair
    s = element[0]
    # Query
    q = element[1]
    # Candidates
    c = element[2]
    # Number of sentences/windows in the context
    story_size = len(s)
    # Number of answers candidates
    candidates_size = len(c)

    # Update the dataset statistics
    max_story_size = max(max_story_size, story_size)
    mean_story_size = (story_size + i*mean_story_size) / (i+1)
    sentence_size = max(sentence_size, max(map(len, chain.from_iterable([s]))))
    query_size = max(query_size, len(q))
    max_candidates_size = max(max_candidates_size, candidates_size)
    mean_candidates_size = (candidates_size + i*mean_candidates_size) / (i+1)

assert max_story_size > 0 and mean_story_size > 0 and sentence_size > 0 and query_size > 0, 'Error during the computation of size params.'

# Maximum memory size: highest number of memory slots allowed per query
if FLAGS.memory_size > 1:  # use only the most recent potential memories
    memory_size = min(FLAGS.memory_size, max_story_size)
else:  # use all potential memories
    memory_size = max_story_size

# Vocabulary size
vocab_size = len(word2idx) + 1  # +1 for nil word

# Log some statistics about the dataset
logger.info("Vocabulary size {}".format(vocab_size))
logger.info("Longest sentence length {}".format(sentence_size))
logger.info("Longest query length {}".format(query_size))
logger.info("Biggest answer candidates set size {}".format(max_candidates_size))
logger.info("Average answer candidates set size {}".format(mean_candidates_size))
logger.info("Longest story length {}".format(max_story_size))
logger.info("Average story length {0:.1f}".format(mean_story_size))

# Highest sentence/query length
sentence_size = max(query_size, sentence_size)
# Size of the answer candidates
candidates_size = max_candidates_size

fs_train, fq_train, fc_train, fa_train = vectorize_data(train, word2idx, sentence_size, memory_size, candidates_size,
                                                        "{}{}train_".format(path, 'filtered_{}/'.format(FLAGS.term_freq_thr) if FLAGS.discard_rare_words else ''))
fs_test, fq_test, fc_test, fa_test = vectorize_data(test, word2idx, sentence_size, memory_size, candidates_size,
                                                    "{}{}test_".format(path, 'filtered_{}/'.format(FLAGS.term_freq_thr) if FLAGS.discard_rare_words else ''))

fs_train = tables.open_file(fs_train, mode='r')
fq_train = tables.open_file(fq_train, mode='r')
fc_train = tables.open_file(fc_train, mode='r')
fa_train = tables.open_file(fa_train, mode='r')

fs_test = tables.open_file(fs_test, mode='r')
fq_test = tables.open_file(fq_test, mode='r')
fc_test = tables.open_file(fc_test, mode='r')
fa_test = tables.open_file(fa_test, mode='r')

# Size of the training set
n_train = len(train)
# Size of the testing set
n_test = len(test)

logger.info("Training Size {}".format(n_train))
logger.info("Testing Size {}".format(n_test))

batch_size = FLAGS.batch_size
batches = zip(range(0, n_train-batch_size, batch_size), range(batch_size, n_train, batch_size)) + \
          [((n_train/batch_size)*batch_size, n_train)]
cbt_test_batches = zip(range(0, (n_test/4)-batch_size, batch_size), range(batch_size, (n_test/4), batch_size)) + \
               [(((n_test/4)/batch_size)*batch_size, (n_test/4))]

if FLAGS.memory_representation == 'sentence':
    encoding = position_encoding
else:
    encoding = identity_encoding

with tf.Graph().as_default():
    # FIXME: When the script is run, the GPU options are not respected (i.e., the script allocates the full GPU memory).
    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.3)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options)

    with tf.Session(config=session_conf) as sess:

        model = MemN2N_KV(memory_representation=FLAGS.memory_representation, window_size=FLAGS.window_size,
                          encoding=encoding,
                          batch_size=batch_size, vocab_size=vocab_size, vocab=word2idx,
                          glove=FLAGS.glove,
                          dropout=FLAGS.dropout,
                          sentence_size=sentence_size, memory_size=memory_size,
                          candidates_size=max_candidates_size,
                          embedding_size=FLAGS.embedding_size,
                          starter_learning_rate=FLAGS.learning_rate,
                          l2=FLAGS.l2, l2_lambda=FLAGS.l2_lambda,
                          gradient_clipping=FLAGS.gradient_clipping, max_grad_norm=FLAGS.max_grad_norm,
                          gradient_noise=FLAGS.add_gradient_noise, anneal_noise=FLAGS.anneal_noise,
                          hops=FLAGS.hops, reader=FLAGS.reader,
                          session=sess,
                          logger=logger)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        saver = tf.train.Saver(max_to_keep=2)

        x = []
        training_accuracies = []
        testing_accuracies = []
        prev_train_acc = 0.0
        timeout = 10
        word_types = {0: 'NE', 1: 'CN', 2: 'V', 3: 'P'}

        for t in range(1, FLAGS.epochs+1):
            # Shuffle the batch order
            np.random.shuffle(batches)

            for start, end in batches:
                # start, end = 0, batch_size
                s = fs_train.root.data[start:end]
                q = fq_train.root.data[start:end]
                c = fc_train.root.data[start:end]  # if FLAGS.data == 'cbt' else candidate_answers
                a = fa_train.root.data[start:end]
                step, summary = model.batch_fit(s, q, c, a, FLAGS.keep_prob)
                writer.add_summary(summary, step)

            if not FLAGS.debug_mode:
                saver.save(sess, log_dir + model.name(), global_step=t)
            # Test on training dataset
            train_preds = []
            for start in range(0, n_train, batch_size):
                end = start + batch_size
                pred, _ = model.predict(fs_train.root.data[start:end],
                                        fq_train.root.data[start:end],
                                        fc_train.root.data[start:end])
                train_preds += list(pred)
            train_labels = [idx for l in fa_train.root.data[:n_train] for idx in l]
            train_acc = metrics.accuracy_score(train_labels, train_preds[:n_train])
            x.append(t)
            training_accuracies.append(train_acc)
            logger.info('-----------------------')
            logger.info('Epoch {}'.format(t))
            logger.info('Training Accuracy: {0:.3f}'.format(train_acc))
            # Test on testing dataset
            if FLAGS.data == 'cbt':
                tmp_acc = []
                for i in range(4):
                    test_preds = []
                    for s, e in cbt_test_batches:
                        start, end = s + (n_test/4)*i, e + (n_test/4)*i
                        pred, _ = model.predict(fs_test.root.data[start:end],
                                                fq_test.root.data[start:end],
                                                fc_test.root.data[start:end])
                        test_preds += list(pred)
                    test_labels = [idx for l in fa_test.root.data[i*(n_test/4):(i+1)*(n_test/4)] for idx in l]
                    test_acc = metrics.accuracy_score(test_labels, test_preds)
                    tmp_acc.append(test_acc)
                    logger.info('Testing Accuracy ({1}): {0:.3f}'.format(test_acc, word_types[i]))
                testing_accuracies.append(tmp_acc)
            else:
                test_preds = []
                for start in range(0, n_test, batch_size):
                    end = start + batch_size
                    pred, _ = model.predict(fs_test.root.data[start:end],
                                            fq_test.root.data[start:end],
                                            fc_test.root.data[start:end])
                    test_preds += list(pred)
                test_labels = [idx for l in fa_test.root.data[:n_test] for idx in l]
                test_acc = metrics.accuracy_score(test_labels, test_preds[:n_test])
                testing_accuracies.append(test_acc)
                logger.info('Testing Accuracy: {0:.3f}'.format(test_acc))
            logger.info('-----------------------')

            # Continue if the training accuracy is higher since last 10 epochs
            if train_acc > prev_train_acc:
                prev_train_acc = train_acc
                timeout = 10
            else:
                if timeout > 0:
                    timeout -= 1
                else:
                    break

        # Close the summary writer
        writer.close()

        # Plot both the training and testing accuracies ans save the figure
        plt.plot(x, [100*y for y in training_accuracies], label='train accuracy')
        if FLAGS.data == 'cbt':
            for i in range(4):
                plt.plot(x, [100*y[i] for y in testing_accuracies], label='test accuracy ({})'.format(word_types[i]))
        else:
            plt.plot(x, [100*y for y in testing_accuracies], label='test accuracy')

        plt.xlabel('Training Epochs')
        plt.ylabel('Accuracy')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.savefig("{}accuracies_{}.jpg".format(log_dir, timestamp))

        # if FLAGS.debug_mode:
        #     # random_batch = list(np.random.choice(n_test, FLAGS.batch_size, replace=False))
        #     random_batch = range(5)
        #
        #     pred, memory_probs = model.predict(fs_test.root.data[random_batch],
        #                                        fq_test.root.data[random_batch],
        #                                        fc_test.root.data[random_batch])
        #     model.display_memory_probs(pred, memory_probs,
        #                                fs_test.root.data[random_batch],
        #                                fq_test.root.data[random_batch],
        #                                fa_test.root.data[random_batch],
        #                                idx2word)

        # Run t-SNE on word representation matrix and visualize it
        # tsne_viz(sess.run(model.A), ['nil_word'] + vocab, "{}tsne_A.svg".format(log_dir))
        # tsne_viz(sess.run(model.B), ['nil_word'] + vocab, "{}tsne_B.svg".format(log_dir))

# Close files where the training set is loaded from
fs_train.close()
fq_train.close()
fc_train.close()
fa_train.close()

# Close files where the testing set is loaded from
fs_test.close()
fq_test.close()
fc_test.close()
fa_test.close()
