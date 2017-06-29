from __future__ import absolute_import

import json
import os
import re
from pprint import pprint
from itertools import chain

from matplotlib import pylab as plt

import nltk
import numpy as np
import tables

from tsne_python.tsne import tsne


def load_task(data_dir, task_id, only_supporting=False):
    """Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    """
    assert 0 < task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def get_stories(f, only_supporting=False):
    """Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    """
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def parse_stories(lines, only_supporting=False):
    """Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.

    Parameters
    ----------
    lines : list of str
        The list of lines from a file that contains one bAbI task.
    only_supporting : bool
        If this is set to True, only the sentences that support the answer are kept.

    Returns
    ----------
    data : list
        The data representing the bAbI task. Each element in this list is a (Story, Query, Answer) tuple.
    """
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:  # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append('')
        else:  # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def load_data(data, training_percentage, testing_percentage, memory_representation, window_size, filter=None):
    """Load the data from its files.

    Parameters
    ----------
    data : str
        The name of the data set on which the model will run.
    training_percentage : float
        The percentage of the training data set to load. This is useful for running the model on a sample data set.
    testing_percentage : float
        The percentage of the testing data set to load.
    memory_representation : str
        The memory representation of the Key-Value Memory Network.
    window_size : int
        The size of a window in case of a window memory.

    Returns
    ----------
    train_data : list
        The loaded training set. Each element in this set is a (Story, Query, Candidates, Answer) tuple.
    test_data : list
        The loaded testing set. Each element in this set is a (Story, Query, Candidates, Answer) tuple.
    """
    prefix = '../../data/'
    data_dir = {'cbt': 'CBTest/data/', 'squad': 'SQuAD/', 'cnn': 'cnn/questions/'}
    # Dictionary that maps each set to its respective size
    cnn_files = {'training': 380298, 'test': 3198}

    if data == 'cbt':
        train_files = [os.path.join(prefix, data_dir[data], "cbtest_{}_train.txt".format(word_type)) for word_type in ['NE', 'CN', 'V', 'P']]
        test_files = [os.path.join(prefix, data_dir[data], "cbtest_{}_test_2500ex.txt".format(word_type)) for word_type in ['NE', 'CN', 'V', 'P']]
    elif data == 'squad':
        train_files = [os.path.join(prefix, data_dir[data], 'train-v1.1.json')]
        test_files = [os.path.join(prefix, data_dir[data], 'dev-v1.1.json')]
    else:
        train_files = [os.path.join(prefix, data_dir[data], 'training/', fn)
                       for fn in sorted(os.listdir(os.path.join(prefix, data_dir[data], 'training/')))]
        test_files = [os.path.join(prefix, data_dir[data], 'test/', fn)
                      for fn in sorted(os.listdir(os.path.join(prefix, data_dir[data], 'test/')))]

    cnn_data_size = int(cnn_files['training'] * training_percentage)
    train_data = []
    for train_file in train_files:
        train_data.extend(get_data(train_file, training_percentage, memory_representation, window_size, filter))
        if data == 'cnn' and len(train_data) >= cnn_data_size:
            break
    cnn_data_size = int(cnn_files['test'] * testing_percentage)
    test_data = []
    for test_file in test_files:
        test_data.extend(get_data(test_file, testing_percentage, memory_representation, window_size, filter))
        if data == 'cnn' and len(test_data) >= cnn_data_size:
            break
    return train_data, test_data


def get_data(filename, percentage, memory_representation, window_size, filter):
    if 'CBTest' in filename:
        return get_cbt_data(filename, percentage, memory_representation, window_size, filter)
    elif 'SQuAD' in filename:
        return get_squad_data(filename, percentage, memory_representation, window_size, filter)
    else:
        return get_cnn_data(filename, memory_representation, window_size, filter)


def get_cbt_data(filename, percentage, memory_representation, window_size, filter):
    """
    Get the CBT data set.

    :param filename: name of the file where the data is saved.
    :param percentage: percentage of the data set to load.
    :param memory_representation: memory representation (i.e., sentence-level or window-level).
    :param window_size: number of words in the window if the memory is window-level.
    :return: data.
    """
    files_size = {'cbtest_NE_train.txt': 67128, 'cbtest_NE_valid_2000ex.txt': 2000, 'cbtest_NE_test_2500ex.txt': 2500,
                  'cbtest_CN_train.txt': 121176, 'cbtest_CN_valid_2000ex.txt': 2000, 'cbtest_CN_test_2500ex.txt': 2500,
                  'cbtest_V_train.txt': 109111, 'cbtest_V_valid_2000ex.txt': 2000, 'cbtest_V_test_2500ex.txt': 2500,
                  'cbtest_P_train.txt': 67128, 'cbtest_P_valid_2000ex.txt': 2000, 'cbtest_P_test_2500ex.txt': 2500}

    data_size = int(files_size[filename.split('/')[-1]] * percentage)

    data = []
    with open(filename) as f:
        story = []
        lines = f.readlines()
        for line in lines:
            line = str.lower(line)
            if line != '\n':
                nid, line = line.split(' ', 1)
                nid = int(nid)
                if nid == 1:
                    story = []
                if '\t' in line:  # query
                    q, a, _, candidates = line.split('\t')
                    q = tokenize(q, data='cbt')
                    if filter is not None:
                        tmp = []
                        for w in q:
                            if w not in filter:
                                tmp.append(w)
                        q = tmp
                    # answer is one vocab word even if it's actually multiple words
                    a = [a]
                    c = [cand.strip() for cand in candidates.split('|') if cand]
                    if filter is not None:
                        tmp = []
                        for w in c:
                            if w not in filter:
                                tmp.append(w)
                        c = tmp
                    if memory_representation == 'sentence':  # sentence-level memory representation
                        # Provide all the substories
                        substory = [x for x in story if x]
                    else:  # window-level memory representation
                        # Window size
                        b = window_size

                        substory = []
                        for sentence in story + [q]:
                            s = (b/2)*[None] + sentence + (b/2)*[None]
                            idx = [i for i, w in enumerate(s) if w in c]
                            if idx:
                                for i in idx:
                                    substory.append(s[i-b/2:i+b/2+1])

                        q = (b/2)*[None] + q + (b/2)*[None]
                        xid = q.index('xxxxx')
                        q = q[xid-b/2:xid+b/2+1]
                    q.remove('xxxxx')

                    data.append((substory, q, c, a))
                    if len(data) >= data_size:
                        return data
                    story.append('')
                else:  # regular sentence
                    sent = tokenize(line, data='cbt')
                    if filter is not None:
                        tmp = []
                        for w in sent:
                            if w not in filter:
                                tmp.append(w)
                        sent = tmp
                    story.append(sent)
    return data


def get_squad_data(filename, percentage, memory_representation, window_size, filter):
    """
    Get the SQuAD data set.

    :param filename:
    :param percentage:
    :param memory_representation:
    :param window_size:
    """
    # Dictionary that maps each set to its respective size
    files = {'train-v1.1.json': 87599, 'dev-v1.1.json': 10570}
    # Expected data size to load
    data_size = int(files[filename.split('/')[-1]] * percentage)

    # Set of stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))

    data = []
    with open(filename) as f:
        # fp: dict(version=unicode, data=list)
        fp = json.load(f)
        # d: dict(paragraphs=list, title=string)
        for d in fp['data']:
            # paragraph: dict(qas=list, context=string)
            for paragraph in d['paragraphs']:
                # Context
                context = paragraph['context']
                # corenlp = StanfordCoreNLP(corenlp_path="./stanford_corenlp_python/stanford-corenlp-full-2014-08-27/")
                # parsed = corenlp.parse(context)
                # pprint(parsed)
                # tree = nltk.tree.Tree.parse(parsed['sentences'][0]['parsetree'])
                # pprint(tree)
                # return 0

                idx2answer = {}
                counter = 0
                # qas: dict(question=string, answers=list)
                for qas in paragraph['qas']:
                    # Answer
                    answer = qas['answers'][0]['text']
                    if answer[-1] == '.':
                        answer = answer[:-1]
                    if ' ' in answer and answer not in idx2answer.values():
                        idx = 'ans' + '0'*(2-len(str(counter))) + str(counter)
                        context = context.replace(answer, idx)
                        idx2answer[idx] = answer
                        counter += 1

                # Story: context split into sentences
                sentences = nltk.sent_tokenize(context)
                story = []
                for sentence in sentences:
                    tokenized = tokenize(sentence, data='squad')
                    corrected = [idx2answer[w] if w in idx2answer else w for w in tokenized]
                    story.append(map(lambda x: x.lower(), corrected))

                # qas: dict(question=string, answers=list)
                for qas in paragraph['qas']:
                    # Query
                    question = qas['question']
                    q = map(lambda x: x.lower(), tokenize(question, data='squad'))
                    # Answer
                    answer = qas['answers'][0]['text'].lower()
                    if answer[-1] == '.':
                        answer = answer[:-1]
                    # Consider the answer as one word
                    a = [answer]
                    if memory_representation == 'sentence':  # sentence-level memory representation
                        substory = [x for x in story]
                    else:  # window-level memory representation
                        substory = []
                        for sentence in story:
                            s = (window_size/2)*[None] + sentence + (window_size/2)*[None]
                            idx = [i for i, w in enumerate(s) if w in q]
                            if idx:
                                for i in idx:
                                    substory.append(s[i-window_size/2:i+window_size/2+1])

                        if len(substory) == 0:
                            continue

                        q = q[-window_size:]

                    # Candidates
                    c = [w for w in list(set(chain.from_iterable(substory))) if w not in stop_words]
                    if None in c:
                        c.remove(None)

                    # Set the candidates same as the answer
                    data.append((substory, q, c, a))
                    if len(data) >= data_size:  # enough elements from the data set
                        return data
    return data


def get_cnn_data(filename, memory_representation, window_size, filter):
    data = []
    with open(filename) as f:
        story = []
        query = []
        answer = []
        entities = {}

        sentences = []
        lines = f.readlines()
        for i, line in enumerate(lines):
            rline = line.rstrip('\n').decode('utf-8')
            if i == 2:
                sentences = nltk.sent_tokenize(rline)
            elif i == 4:
                query = tokenize(rline.lower(), data='cnn')
                if filter is not None:
                    tmp = []
                    for w in query:
                        if w not in filter:
                            tmp.append(w)
                    query = tmp
            elif i == 6:
                answer = rline
            elif i >= 8:
                kv = rline.split(':', 1)
                key, value = kv[0], kv[1].lower()
                entities[key] = value
        c = entities.values()

        if memory_representation == "sentence":
            for s in sentences:
                tmp = tokenize(s.lower(), data='cnn')
                story.append(map(lambda w: entities[w] if w in entities else w, tmp))
        else:
            for s in sentences:
                tmp = (window_size/2)*[None] + tokenize(s.lower(), data='cnn') + (window_size/2)*[None]
                idx = [i for i, w in enumerate(tmp) if w in entities.keys()]
                if idx:
                    for i in idx:
                        story.append(map(lambda w: entities[w] if w in entities else w, tmp[i-window_size/2:i+window_size/2+1]))

            query = (window_size/2)*[None] + query + (window_size/2)*[None]
            xid = query.index('@placeholder')
            query = query[xid-window_size/2:xid+window_size/2+1]
        query.remove('@placeholder')
        q = map(lambda w: entities[w] if w in entities else w, query)

        if filter is not None:
            tmp = []
            for s in story:
                tmp_s = []
                for w in s:
                    if w not in filter:
                        tmp_s.append(w)
                if len(tmp_s) > 0:
                    tmp.append(tmp_s)
            story = tmp

        data.append((story, q, c, [entities[answer]]))
    return data


def tokenize(sent, data='babi'):
    """Return the tokens of a sentence removing symbols.
    """
    if data in ['cbt', 'cnn']:
        return [x for x in sent.split(' ') if any(c.isalnum() for c in x)]
    elif data == 'squad':
        # entity_names = extract_entity_names(nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent)), binary=True))
        # s = sent
        # for e in entity_names:
        #     s = s.replace(e, e.replace(' ', ''))
        return [w for w in nltk.word_tokenize(sent) if any(c.isalnum() for c in w)]
    else:
        return filter(lambda x: x.isalnum(), [x.strip() for x in re.split('(\W+)?', sent) if x.strip()])


def vectorize_data(data, word_idx, sentence_size, memory_size, candidates_size, path):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.

    The vectorized data is written into files in order to reduce the memory usage.
    """
    full_path = path.replace('xxx', 'arrays', 1)

    if all(os.path.isfile("{}{}.h5".format(full_path, filename)) for filename in ['S', 'Q', 'C', 'A']):
        return "{}S.h5".format(full_path), "{}Q.h5".format(full_path), "{}C.h5".format(full_path), "{}A.h5".format(full_path)

    fs = tables.open_file("{}S.h5".format(full_path), mode='w')
    fq = tables.open_file("{}Q.h5".format(full_path), mode='w')
    fc = tables.open_file("{}C.h5".format(full_path), mode='w')
    fa = tables.open_file("{}A.h5".format(full_path), mode='w')
    atom = tables.Int64Atom()

    S = fs.create_earray(fs.root, 'data', atom, (0, sentence_size * memory_size))
    Q = fq.create_earray(fq.root, 'data', atom, (0, sentence_size))
    C = fc.create_earray(fc.root, 'data', atom, (0, candidates_size if candidates_size else 1))  # set a symbolic shape if there is no candidate
    A = fa.create_earray(fa.root, 'data', atom, (0, 1))

    for element in data:
        if not candidates_size:
            story, query, answer = element
        else:
            story, query, candidates, answer = element
            lc = max(0, candidates_size - len(candidates))
            c = [word_idx[c] for c in candidates] + [0] * lc
            C.append(np.array([c]))
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] if w is not None else 0 for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[-memory_size:]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] if w is not None else 0 for w in query] + [0] * lq

        y = [word_idx[a] for a in answer]

        S.append(np.array([list(chain.from_iterable(ss))]))
        Q.append(np.array([q]))
        A.append(np.array([y]))

    fs.close()
    fq.close()
    fc.close()
    fa.close()

    return fs.filename, fq.filename, fc.filename, fa.filename


def tsne_viz(X, vocab, output_filename, colors=None, no_dims=2, initial_dims=50, perplexity=30.0):
    assert X.shape[0] == len(vocab), "Error: X and vocab must have same dimensions."

    if colors is None:
        colors = ['black' for _ in range(len(X))]
    # Run t-SNE on the word representation matrix
    Y = tsne(X, no_dims, initial_dims, perplexity)
    # Plotting:
    xvals, yvals = Y[:, 0], Y[:, 1]
    plt.figure(figsize=(100, 100))
    plt.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        plt.annotate(word, (x, y), fontsize=0.1, color=color)
    plt.savefig(output_filename, bbox_inches='tight', format="svg", dpi=1200)
