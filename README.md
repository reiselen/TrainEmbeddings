# TrainEmbeddings
## Overview
 A simple python wrapper to train embeddings in five different embedding and LM architectures, specifically:
 - [word2vec](https://github.com/RaRe-Technologies/gensim)
 - [GloVe](https://github.com/stanfordnlp/GloVe)
 - [fastText](https://github.com/facebookresearch/fastText)
 - [FLAIR](https://github.com/flairNLP/flair)
 - [RoBERTa](https://github.com/huggingface/transformers)
 - [ELMO](https://github.com/allenai/bilm-tf)

The wrapper is currently designed to train multiple models for multiple languages based on a config file. Example config files are available in the [example_configs](https://github.com/reiselen/TrainEmbeddings/tree/main/example_configs) folder.

## Requirements
Ensure that all required libraries for the different architectures are installed.

`pip install -r requirements.txt`

## Usage
Run the python wrapper specifying the specific config with the relevant settings. More details on specific configuration settings are provided below

`python trainer_cli.py -c ./example_configs/af_config_linux.json`

## Configuration settings
The following list most of the most important settings that can be set in the configuration
All values available to the respective models can be found on the main sites for each of the algorithms, and additional settings can be added to the configuration as long as the parameter name is identical to those specified in the architecture documentation.

### Main options
    "input_dir": Directory where the training data is located, in language specific folders. Training data should be raw utf8 text files,
    "output_dir": Directory where the training output and eventual models will be saved,
    "languages": One or more languages for which embeddings will be trained. The language abbreviations are used to select a specific input directory and name output files,
    "algorithms": One or more algorithms to train. The list of available algorithms is specified below. For each algorithm listed in this option, at least some of the available parameter options should be included in the configuration file.

### [w2v-skipgrams](https://github.com/RaRe-Technologies/gensim)
Skipgram model for word2vec algorithm.
```
    "vector_size": Vector dimensions (int),
    "window": Context window (int),
    "min_count": Minimum word occurence to be included in model (int),
    "workers": Worker threads to use during training (int),
    "epochs": Number of epochs for training (int),
    "sg": Train skipgram model (0=cbow, 1=skipgram),
```

### [w2v-cbow](https://github.com/RaRe-Technologies/gensim)
Continuous bag of words model for word2vec algorithm.
```
    "vector_size": Vector dimensions (int),
    "window": Context window (int),
    "min_count": Minimum word occurence to be included in model (int),
    "workers": Worker threads to use during training (int),
    "epochs": Number of epochs for training (int),
    "sg": Train skipgram model (0=cbow, 1=skipgram),
```
### [glove](https://github.com/stanfordnlp/GloVe)
```
    "verbose": Amount of information to print (int),
    "memory": Memory in Gb to use during training (float),
    "vocab_min_count": Minimum word occurence to be included in model (int),
    "vector_size": Vector dimensions (int),
    "max_iter": Number of epochs for training (int),
    "window_size": Context window size (int),
    "num_threads": Worker threads to use during training (int)
```
### [fasttext-s](https://fasttext.cc/docs/en/options.html)
```
    "model": "skipgram",
    "lr": Learning rate (float),
    "dim": Vector dimensions (int),
    "ws": Context window size (int),
    "epoch": Number of epochs for training (int),
    "minCount": Minimum word occurence to be included in model (int),
    "minn": Minimum length of subword n-gram (int),
    "maxn": Maximum length of subword n-grams (int),
    "neg": Number of negatives sampled (int),
    "wordNgrams": Number of words to include in ngrams (int),
    "loss": Loss function (ns, hs, softmax),
    "bucket": Number of buckets,
    "thread": Worker threads to use during training (int),
```
### [fasttext-c](https://fasttext.cc/docs/en/options.html)
```
    "model": "cbow",
    "lr": Learning rate (float),
    "dim": Vector dimensions (int),
    "ws": Context window size (int),
    "epoch": Number of epochs for training (int),
    "minCount": Minimum word occurence to be included in model (int),
    "minn": Minimum length of subword n-gram (int),
    "maxn": Maximum length of subword n-grams (int),
    "neg": Number of negatives sampled (int),
    "wordNgrams": Number of words to include in ngrams (int),
    "loss": Loss function (ns, hs, softmax),
    "bucket": Number of buckets,
    "thread": Worker threads to use during training (int),
```
### [flair-f](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)
```
    "temp_file_sent_size": File size of temp files (int),
    "hidden_size": LSTM hidden size (int),
    "nlayers": Number of layers (int),
    "sequence_length": Context length (int),
    "mini_batch_size": Batch size (int),
    "learning_rate": Learning rate (float),
    "epochs": Number of training epochs,
    "is_forward_lm": true,
    "character_level": Is character level (bool),
    "pretrained_model": Path to an existing pretrained model (str)
```
### [flair-b](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_9_TRAINING_LM_EMBEDDINGS.md)
```
    "temp_file_sent_size": File size of temp files (int),
    "hidden_size": LSTM hidden size (int),
    "nlayers": Number of layers (int),
    "sequence_length": Context length (int),
    "mini_batch_size": Batch size (int),
    "learning_rate": Learning rate (float),
    "epochs": Number of training epochs,
    "is_forward_lm": false,
    "character_level": Is character level (bool),
    "pretrained_model": Path to an existing pretrained model
```
### [elmo](https://github.com/allenai/bilm-tf)
Elmo has an entire separate jsonett file specifying the specific configuration. This can be found in [trainers/configs](https://github.com/reiselen/TrainEmbeddings/tree/main/trainers/configs)
```
    "config_file": "./trainers/configs/bidirectional_language_model.jsonnet"
```
### [roberta](https://github.com/huggingface/transformers)
```
    "roberta_config": {
        "vocab_size": Size of the vocabulary for BPE (int),
        "max_position_embeddings": Max position embeddings (int),
        "hidden_size": Hidden size of Transformer (int),
        "num_attention_heads": Number of attention heads (int),
        "num_hidden_layers": Number of hidden layers (int),
        "type_vocab_size": Minimum word occurence for inclusion in vocab (int)
    },
    "epochs": Number of training epochs (int),
    "model": Path to existing pretrained model (str),
    "lines_per_instance": Number of lines to process in single iteration (int),
    "mask_percentage": Percentage of tokens to mask during training (float),
    "batch_size": Batch size (int),
    "max_position_size": Maximum context length (int),
	"cuda_instance": Specific GPU to use during training (int)
```
## License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use these file except in compliance with the License.

You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.