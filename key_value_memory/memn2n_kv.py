from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf
from six.moves import range


def get_glove(embedding_size, vocab, logger):
    glove_filename = "glove.6B/glove.6B.{}d.txt".format(embedding_size)
    array_dtype = [('word', '|S25')] + [('e' + str(i), 'f4') for i in range(1, embedding_size+1)]
    array = np.loadtxt(glove_filename, dtype=array_dtype, comments=None)
    indices, updates = [], []
    for elem in array:
        w = elem[0]
        if w in vocab:
            indices.append(vocab[w])
            updates.append([elem[j] for j in range(1, embedding_size+1)])
    logger.info("{} / {} words have a vector representation in GloVe.".format(len(indices), len(vocab)))
    return indices, updates


def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]

    It captures the order of the words in the sentence.
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def identity_encoding(sentence_size, embedding_size):
    return np.ones((sentence_size, embedding_size), dtype=np.float32)


def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros([1, s])
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)


def zero_indices_slot(t, indices, name=None):
    """
    Overwrites rows that are initialized with pre-trained GloVe embeddings of the input Tensor with zeros.
    Also overwrites the nil_slot (first row) with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t, indices]) as name:
        t = tf.convert_to_tensor(t, name="t")
        v = tf.Variable(t, trainable=False, name='tmp')
        s = tf.shape(t)[1]
        z = tf.zeros([len(indices), s])
        # result = tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)
        # for i in indices:
        #     result = tf.concat(axis=0, values=[tf.slice(result, [0, 0], [i, -1]),
        #                                        z,
        #                                        tf.slice(result, [i+1, 0], [-1, -1])])
        return tf.scatter_update(v, indices, z)


def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def get_probs_candidates(probabilities, candidates):
    probs_candidates = []
    for i, x in enumerate(candidates):
        probs_candidates.append([probabilities[i][w] if w else np.float32(0) for w in x])
    return np.array(probs_candidates)


def get_argmax_candidates(candidates, probs_candidates):
    argmax_candidates = []
    for i, x in enumerate(np.argmax(probs_candidates, 1)):
        argmax_candidates.append(candidates[i][x])
    return np.array(argmax_candidates)


class MemN2N_KV(object):
    """Key Value Memory Network."""
    def __init__(self, memory_representation, window_size, encoding,
                 batch_size, vocab_size, vocab,
                 glove,
                 dropout,
                 sentence_size, memory_size,
                 candidates_size, embedding_size,
                 starter_learning_rate=0.001,
                 l2=False, l2_lambda=0.2,
                 gradient_clipping=False, max_grad_norm=40.0,
                 gradient_noise=False, anneal_noise=False,
                 hops=3,
                 reader='bow',
                 session=tf.Session(),
                 logger=logging.getLogger(__name__),
                 name='KeyValueMemN2N'):
        """Creates an Key Value Memory Network

        Args:
        memory_representation: How the memory is represented (i.e., sentence-level or window-level).

        batch_size: The size of the batch.

        vocab_size: The size of the vocabulary (should include the nil word). The nil word one-hot encoding should be 0.

        query_size: largest number of words in question

        sentence_size: The max size of a sentence in the data. All sentences should be padded to this length. If padding
        is required it should be done with nil one-hot encoding (0).

        embedding_size: The size of the word embedding.

        memory_size: The max size of the memory.
        
        feature_size: dimension of feature extraced from word embedding

        hops: The number of hops. A hop consists of reading and addressing a memory slot.

        candidates: If it is True, then each query comes with output candidates.

        session: Tensorflow Session the model is run with. Defaults to `tf.Session()`.

        name: Name of the End-To-End Key-Value Memory Network. Defaults to `KeyValueMemN2N`.
        """
        self._memory_representation = memory_representation
        self._window_size = window_size
        self._center_id = int(self._window_size / 2)
        self._glove = glove
        self._dropout = dropout
        self._vocab = vocab
        self._sentence_size = sentence_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._candidates_size = candidates_size
        self._memory_size = memory_size
        self._key_size = self._sentence_size
        self._value_size = self._sentence_size if self._memory_representation == 'sentence' else 1
        self._embedding_size = embedding_size
        self._starter_learning_rate = starter_learning_rate
        self._l2 = l2
        self._l2_lambda = l2_lambda
        self._gradient_clipping = gradient_clipping
        self._max_grad_norm = max_grad_norm
        self._gradient_noise = gradient_noise
        self._anneal_noise = anneal_noise
        self._hops = hops
        self._sess = session
        self._logger = logger
        self._name = name
        self._reader = reader

        self._build_inputs()
        self._build_vars()

        if self._glove:
            self.indices, self.updates = get_glove(self._embedding_size, self._vocab, self._logger)
            # self.indices = tf.constant(indices, dtype=tf.int32, name="glove_indices")
            # self.updates = tf.constant(updates, dtype=tf.float32, name="glove_updates")

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Shape: [batch_size, sentence_size, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.A, self._queries)
            # Shape: [batch_size, memory_size, key_size, embedding_size]
            self.mkeys_embedded_chars = tf.nn.embedding_lookup(self.A, self._memory_keys)
            # Shape: [batch_size, memory_size, value_size, embedding_size]
            self.mvalues_embedded_chars = tf.nn.embedding_lookup(self.A, self._memory_values)

        # Decay learning rate
        self._learning_rate = tf.train.exponential_decay(self._starter_learning_rate, self.global_step, 20000, 0.96, staircase=True)
        # Adam optimizer
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate, epsilon=0.1)

        # Shape: [sentence_size, embedding_size]
        self._encoding = tf.constant(encoding(self._sentence_size, self._embedding_size), name="encoding")

        if reader == 'bow':
            # Shape: [batch_size, embedding_size]
            q_r = tf.reduce_sum(self.embedded_chars*self._encoding, 1, name='q_r')
            # Shape: [batch_size, memory_size, embedding_size]
            doc_r = tf.reduce_sum(self.mkeys_embedded_chars*self._encoding, 2, name='doc_r')
            # Shape: [batch_size, memory_size, embedding_size]
            value_r = tf.reduce_sum(self.mvalues_embedded_chars*self._encoding, 2, name='value_r')
        elif reader == 'simple_gru':
            x_tmp = tf.reshape(self.mkeys_embedded_chars, [-1, self._sentence_size, self._embedding_size])
            x = tf.transpose(x_tmp, [1, 0, 2])
            # Reshape to (n_steps*batch_size, n_input)
            x = tf.reshape(x, [-1, self._embedding_size])
            # Split to get a list of 'n_steps'
            # tensors of shape (doc_num, n_input)
            x = tf.split(axis=0, num_or_size_splits=self._sentence_size, value=x)

            # do the same thing on the question
            q = tf.transpose(self.embedded_chars, [1, 0, 2])
            q = tf.reshape(q, [-1, self._embedding_size])
            q = tf.split(axis=0, num_or_size_splits=self._sentence_size, value=q)

            k_rnn = tf.contrib.rnn.GRUCell(self._embedding_size)
            q_rnn = tf.contrib.rnn.GRUCell(self._embedding_size)

            with tf.variable_scope('story_gru'):
                doc_output, _ = tf.contrib.rnn.static_rnn(k_rnn, x, dtype=tf.float32)
            with tf.variable_scope('question_gru'):
                q_output, _ = tf.contrib.rnn.static_rnn(q_rnn, q, dtype=tf.float32)
                doc_r = tf.nn.dropout(tf.reshape(doc_output[-1], [-1, self._memory_size, self._embedding_size]), self.keep_prob)
                value_r = doc_r
                q_r = tf.nn.dropout(q_output[-1], self.keep_prob)

        with tf.name_scope('cross_entropy'):
            logits, memory_op = self._inference(doc_r, value_r, q_r)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(self._labels, tf.float32), name='cross_entropy')
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        with tf.name_scope('loss_op'):
            loss_op = cross_entropy_sum
            # L2 regularization
            if self._l2:
                vars = tf.trainable_variables()
                lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
                loss_op += l2_lambda*lossL2

        with tf.name_scope('gradient_pipeline'):
            grads_and_vars = self._optimizer.compute_gradients(loss_op)
            if self._gradient_clipping:
                grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g, v in grads_and_vars]
            if self._gradient_noise:
                stddev = tf.sqrt(tf.realdiv(1.0, tf.pow(tf.cast(tf.add(1, self.global_step ), tf.float32), 0.55))) if self._anneal_noise else 1e-3
                grads_and_vars = [(add_gradient_noise(g, stddev=stddev), v) for g, v in grads_and_vars]
            nil_grads_and_vars = []
            for g, v in grads_and_vars:
                if v.name in self._nil_vars:
                    # if self._glove:
                    #     nil_grads_and_vars.append((zero_indices_slot(g, self.indices), v))
                    # else:
                    nil_grads_and_vars.append((zero_nil_slot(g), v))
                else:
                    nil_grads_and_vars.append((g, v))
            # Apply the specified gradients to trainable variables. Also increment the global step
            train_op = self._optimizer.apply_gradients(nil_grads_and_vars, name="train_op", global_step=self.global_step)

        with tf.name_scope('predict_ops'):
            # Shape: [batch_size, vocabulary_size]
            probs = tf.nn.softmax(logits)
            # probs = tf.Print(probs, ["probs", probs[0], probs[0][self._candidates[0][5]], tf.shape(probs)], summarize=100)
            if self._candidates_size > 0:
                # Shape: [batch_size, candidates_size]
                probs_candidates = tf.py_func(get_probs_candidates, [probs, self._candidates], tf.float32)
                # probs_candidates = tf.Print(probs_candidates, ["probs_c", probs_candidates[0], tf.shape(probs_candidates)], summarize=100)
                # Shape: [batch_size]
                predict_op = tf.py_func(get_argmax_candidates, [self._candidates, probs_candidates], tf.int32, name="predict_op")
                # predict_op = tf.Print(predict_op, ["predict_op", predict_op[0], tf.argmax(self._labels[0]), self._candidates[0], tf.shape(predict_op)], summarize=100)
            else:
                predict_op = tf.argmax(probs, 1, name="predict_op")

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.cast(tf.argmax(self._labels, 1), tf.int32), predict_op)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.train_op = train_op
        self.memory_op = memory_op

        # Create summaries
        tf.summary.scalar("global_step", self.global_step)
        tf.summary.scalar("learning_rate", self._learning_rate)
        tf.summary.scalar('cross_entropy_sum', cross_entropy_sum)
        tf.summary.scalar('cross_entropy_mean', cross_entropy_mean)
        tf.summary.scalar('loss_op', self.loss_op)
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries into a single "operation" which we can execute in a session
        self.summary_op = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

        if self._glove:
            with tf.variable_scope(self._name, reuse=True):
                # print(session.run(self.W[self.indices[0]]))
                _ = self._sess.run(tf.scatter_update(self.A, self.indices, self.updates))
                _ = self._sess.run(tf.scatter_update(self.B, self.indices, self.updates))
                # print(session.run(tf.equal(self.updates[0], self.W[self.indices[0]])))

    def _build_inputs(self):
        with tf.name_scope("inputs"):
            # shape: [batch_size, memory_size, key_size]
            self._memory_keys = tf.placeholder(tf.int32, [None, self._memory_size, self._key_size], name='memory_keys')
            # shape: [batch_size, memory_size, value_size]
            self._memory_values = tf.placeholder(tf.int32, [None, self._memory_size, self._value_size], name='memory_values')
            # shape: [batch_size, query_size]
            self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name='queries')
            # shape: [batch_size, candidates_size]
            self._candidates = tf.placeholder(tf.int32, [None, self._candidates_size], name='candidates')
            # shape: [batch_size, vocab_size]
            self._labels = tf.placeholder(tf.float32, [None, self._vocab_size], name='answers')
            # Probability of outputting the input node
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            A = tf.concat(axis=0, values=[nil_word_slot, tf.contrib.layers.xavier_initializer()([self._vocab_size-1, self._embedding_size]) ])
            B = tf.concat(axis=0, values=[nil_word_slot, tf.contrib.layers.xavier_initializer()([self._vocab_size-1, self._embedding_size]) ])

            self.A = tf.Variable(A, name='A')
            self.B = self.A  # tf.Variable(B, name='B')

            self.r_list = []
            for _ in range(self._hops):
                # Define R for variables
                R = tf.get_variable('R{}'.format(_), shape=[self._embedding_size, self._embedding_size],
                                    initializer=tf.contrib.layers.xavier_initializer())
                self.r_list.append(R)

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self._nil_vars = set([self.A.name, self.B.name])

    '''
    mkeys: the vector representation for keys in memory
    -- shape of each mkeys: [memory_size, embedding_size]
    mvalues: the vector representation for values in memory
    -- shape of each mvalues: [memory_size, embedding_size]
    questions: the vector representation for the question
    -- shape of questions: [1, embedding_size]
    -- shape of R: [feature_size, feature_size]
    -- shape of self.A: [feature_size, embedding_size]
    -- shape of self.B: [feature_size, embedding_size]
    self.A, self.B and R are the parameters to learn
    '''
    def _inference(self, mkeys, mvalues, questions):
        
        with tf.variable_scope(self._name):
            # Relevance probability of each memory slot to the query per hop
            memory_probs = []
            u = [questions]
            for _ in range(self._hops):
                R = self.r_list[_]
                # Shape: [batch_size, 1, embedding_size]
                u_expanded = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                # u_expanded = tf.Print(u_expanded, ["u_expanded_0", u_expanded[0], tf.shape(u_expanded)], summarize=100, name="u_expanded_0")

                # Key addressing
                # Shape: [batch_size, memory_size]
                dotted = tf.reduce_sum(mkeys * u_expanded, 2)
                # dotted = tf.Print(dotted, ["dotted_0", dotted[0], tf.shape(dotted)], summarize=100, name="dotted_0")

                # Calculate probabilities
                # Shape: [batch_size, memory_size]
                probs = tf.nn.softmax(dotted)
                # probs = tf.Print(probs, ["probs_0", probs[0], tf.reduce_sum(probs[0]), tf.shape(probs)], summarize=100, name="probs_0")
                memory_probs.append(probs)
                # Shape: [batch_size, 1, memory_size]
                probs_expanded = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                # probs_expanded = tf.Print(probs_expanded, ["probs_expanded_0", probs_expanded[0], tf.shape(probs_expanded)], summarize=100, name="probs_temp")

                # Value reading
                # Shape: [batch_size, embedding_size, memory_size]
                v_temp = tf.transpose(mvalues, [0, 2, 1])
                # v_temp = tf.Print(v_temp, ["v_temp_0", v_temp[0], tf.shape(v_temp)], summarize=100, name="v_temp")
                # Shape: [batch_size, embedding_size]
                o_k = tf.reduce_sum(v_temp * probs_expanded, 2)
                # o_k = tf.Print(o_k, ["o_k_0", o_k[0], tf.shape(o_k)], summarize=100, name="o_k")
                # Shape: [embedding_size, batch_size]
                u_k = tf.matmul(R, u[-1] + o_k, transpose_b=True)
                # u_k = tf.Print(u_k, ["u_k_0", u_k[0], tf.shape(u_k)], summarize=100, name="u_k")

                u.append(tf.transpose(u_k))

            # Shape: [batch_size, embedding_size]
            o = u[-1]
            # o = tf.Print(o, ["o_0", o[0], tf.shape(o)], summarize=100, name="o")
            # Shape: [batch_size, vocab_size]
            logits = tf.matmul(o, self.B, transpose_b=True)  # + logits_bias
            # logits = tf.Print(logits, ["logits_0", logits[0], tf.shape(logits)], summarize=100, name="logits")
            # logits = tf.nn.dropout(tf.matmul(o, self.B) + logits_bias, self.keep_prob)

            return logits, memory_probs

    def batch_fit(self, stories, queries, candidates, answers, keep_prob):
        mk = np.array([zip(*[iter(context)] * self._sentence_size) for context in stories])
        if self._memory_representation == 'sentence':
            mv = mk[:]
        else:
            mv = []
            for context in mk:
                values = []
                for window in context:
                    values.append([window[self._center_id]])
                mv.append(values)
            mv = np.array(mv)

        l = np.zeros((answers.shape[0], self._vocab_size))
        for i, j in enumerate(answers):
            l[i][j] = 1

        feed_dict = {
            self._queries: queries,
            self._memory_keys: mk,
            self._memory_values: mv,
            self._candidates: candidates,
            self._labels: l,
            self.keep_prob: keep_prob
        }
        _, step, summary = self._sess.run([self.train_op, self.global_step, self.summary_op], feed_dict)
        return step, summary

    def predict(self, stories, queries, candidates):
        mk = np.array([zip(*[iter(context)] * self._sentence_size) for context in stories])
        if self._memory_representation == 'sentence':
            mv = mk[:]
        else:
            mv = []
            for context in mk:
                values = []
                for window in context:
                    values.append([window[self._center_id]])
                mv.append(values)
            mv = np.array(mv)

        feed_dict = {
            self._queries: queries,
            self._memory_keys: mk,
            self._memory_values: mv,
            self._candidates: candidates,
            self.keep_prob: 1.0
        }
        return self._sess.run([self.predict_op, self.memory_op], feed_dict)

    def display_memory_probs(self, prediction, memory_probs, memories, query, answer, idx2word, candidates=None):
        """Displays the memory keys and their probability of relevance to the query at each hop.

        Display the probability of relevance to the query of
        each memory slot at each hop. The display goes one by one from the batch.

        Args:
            memory_probs: The list of the relevance probability of memory keys to the query.
        """
        mk = np.array([zip(*[iter(context)] * self._sentence_size) for context in memories])

        for b in range(len(prediction)):
            q = query[b]
            while not q[-1]:
                q = q[:-1]
            a = answer[b][0]
            print("\nQuery: {}\nAnswer: {}\nPrediction: {}\n".format(' '.join(map(lambda w: idx2word[w], q)),
                                                                     idx2word[a],
                                                                     idx2word[prediction[b]]))
            header = "Memory slots                                                                                     "
            slot_space = len(header)
            for _ in range(self._hops):
                header += "    Hop {}".format(_ + 1)
            print(header)
            for s in range(self._memory_size):
                # The `s`th memory slot
                memory_slot = mk[b, s]
                if np.sum(memory_slot) > 0:
                    while not memory_slot[-1]:
                        memory_slot = memory_slot[:-1]
                    line = ' '.join(map(lambda w: idx2word[w], memory_slot))
                    line += ' ' * max(0, (slot_space - len(line)))
                    for h in range(self._hops):
                        line += "    {0:.3f}".format(memory_probs[h][b][s])
                    print(line)
            raw_input("\nPress ENTER to continue.")

    def name(self):
        return self._name
