# Key Value Memory Networks

This repository contains the implementation of [Key Value Memory Networks for Directly Reading Documents](https://arxiv.org/abs/1606.03126) in Python using Tensorflow. The model is tested on [The Children's Book Test](https://arxiv.org/abs/1511.02301) and [CNN QA](http://cs.nyu.edu/~kcho/DMQA/).

![Structure of Key Value Memory Networks](key_value_mem.png)

### Get Started

```
git clone https://github.com/canguezelhan/key-value-memory-network.git

mkdir ./key-value-memory-network/key_value_memory/logs
mkdir ./key-value-memory-network/key_value_memory/data/
cd ./key-value-memory-network/key_value_memory/data
wget http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz
tar xzvf ./CBTest.tgz
wget https://drive.google.com/uc?export=download&id=0BwmD_VLjROrfTTljRDVZMFJnVWM
tar xzvf ./cnn.tgz

cd ../
python model.py
```

### Usage

#### Running model.py

The script, when executed, creates several files as output. These files can be found in the directory key_value_memory/results0/

There are serval flags within model.py. Check all avaiable flags with the following command.
```
python model.py -h
```

### Requirements

* tensorflow
* scikit-learn
* six
* nltk
