# Copyright 2023 Centre for Text Technology, North-West University. 
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum

from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import (WordEmbeddings, FlairEmbeddings, 
                TokenEmbeddings, FastTextEmbeddings, BertEmbeddings, ELMoEmbeddings)
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from GloveEmbeddings import GloveEmbeddings
from pathlib import Path

import flair
import torch

class EmbeddingType(Enum):
    GLOVE=1
    FLAIR=2
    FASTTEXT=3
    ELMO=4
    ROBERTA=5
    W2V=6

    def text_to_Value(textValue: str):
        val_to_test = textValue.lower()
        if (val_to_test == 'g'):
            return EmbeddingType.GLOVE
        elif(val_to_test == 'f'):
            return EmbeddingType.FLAIR
        elif(val_to_test == 'ft'):
            return EmbeddingType.FASTTEXT
        elif(val_to_test == 'bert'):
            return EmbeddingType.ROBERTA
        elif(val_to_test == 'elmo'):
            return EmbeddingType.ELMO
        elif(val_to_test == 'w2v'):
            return EmbeddingType.W2V

class TrainLabeller:
    hidden_size=512
    use_crf=True
    train_with_dev=False
    max_epochs=40
    mini_batch_size=64
    learn_rate = 0.1
    columns = {0: 'Token', 1: 'Tag'}
    cuda_available = torch.cuda.is_available()

    def train_tagger(self, train_file: Path, test_file: Path, embedding_path: Path,
                        embedding_type: EmbeddingType, cuda_instance: int = -1) -> float:
        """
        Wrapper class to Train a FLAIR sequence labeller with a single embedding
        to act as downstream verification of the quality of the embedding
        for the purpose of tuning. Returns the metric value of the training run

        # Parameters
        train_file : The file with the training data that will be used to train FLAIR labeller
        test_file : The test file with testing data to generate the evaluation metric
        embedding_path : `Path` object to the embedding that should be loaded
        embedding_type: `EmbeddingType` enum value for the type of embedding being loaded.
                        This should correspond to the type of embedding in the embedding_path
        cuda_instance : The gpu instance which should be used to train the FLAIR labeller. 
                        If not specified, training will happen on the CPU (and be VERY slow)
        """
        # Passing the relevant GPU instance to FLAIR
        if (self.cuda_available
            and
            cuda_instance > 0):
            flair.device = torch.device('cuda', cuda_instance)
            
        # Load the training and testing data
        corpus: Corpus = ColumnCorpus(train_file.parent, self.columns, 
                                train_file=train_file.name, test_file=test_file.name)
        tag_dictionary = corpus.make_label_dictionary(label_type=self.columns[1])
        print(embedding_path)

        # Load the specified embedding
        embeddings = self.__load_embedding__(embedding_path, embedding_type)
        if (not embeddings):
            # The specified embedding couldn't be loaded, return negative score
            # rather than failing and possibly interupting tuning
            return -1

        # Init a simple biLSTM sequence labeller with the loaded embeddings
        tagger: SequenceTagger = SequenceTagger(hidden_size=self.hidden_size,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=self.columns[1],
                                                use_crf=self.use_crf)

        # 6. initialize trainer
        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        # 7. start training
        result = trainer.train('./temp_models/',
                    learning_rate=self.learn_rate,
                    mini_batch_size=self.mini_batch_size,
                    max_epochs=self.max_epochs, train_with_dev=self.train_with_dev, 
                    save_final_model=False, save_model_each_k_epochs=-1, param_selection_mode=True)
        return result['test_score']

    def __load_embedding__(self, model_file: Path, embed_type: EmbeddingType) -> TokenEmbeddings:
        if (not model_file.exists()):
            return None
        if (embed_type == EmbeddingType.GLOVE):
            return GloveEmbeddings(str(model_file))
        elif (embed_type == EmbeddingType.FASTTEXT):
            return FastTextEmbeddings(str(model_file))
        elif (embed_type == EmbeddingType.FLAIR):
            return FlairEmbeddings(str(model_file))
        elif (embed_type == EmbeddingType.ROBERTA):
            return BertEmbeddings(str(model_file))
        elif (embed_type == EmbeddingType.ELMO):
            return ELMoEmbeddings(str(model_file))
        elif (embed_type == EmbeddingType.W2V):
            return WordEmbeddings(str(model_file))
        else:
            return None
