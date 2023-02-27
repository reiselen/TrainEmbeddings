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

from collections import Counter
from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl

from flair.data import Dictionary
from flair.embeddings import FlairEmbeddings
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus

from pathlib import Path

from ray.tune.syncer import Syncer
from ray.tune import result

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from utils.trainer import Trainer, TrainingConfig

import flair
import pickle
import regex as re
import shutil
import torch

if TYPE_CHECKING:
    from ray.tune.logger import Logger


class flairConfig(TrainingConfig):
    """
    Training configuration class for FLAIR embeddings
    """
    temp_file_sent_size:int = 100000,
    hidden_size:int = 2048,
    nlayers:int = 1,
    sequence_length:int = 250,
    mini_batch_size:int = 128,
    learning_rate:int = 10,
    epochs:int = 15,
    is_forward_lm:bool = True,
    character_level:bool = True
    pretrained_model: str = None
    num_workers:int = 1

    def __init__(
            self,
            temp_file_sent_size:int = 100000,
            hidden_size:int = 2048,
            nlayers:int = 1,
            sequence_length:int = 250,
            mini_batch_size:int = 128,
            learning_rate:int = 10,
            epochs:int = 15,
            is_forward_lm:bool = True,
            character_level:bool = True,
            num_workers:int = 1
            ):
        self.temp_file_sent_size=temp_file_sent_size
        self.hidden_size=hidden_size
        self.nlayers=nlayers
        self.sequence_length=sequence_length
        self.mini_batch_size=mini_batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.is_forward_lm=is_forward_lm
        self.character_level=character_level
        self.num_workers = num_workers

    def get_config_from_dict(config_dict: Dict[str, Any] = None) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object specifically for FLAIR that can be reused by the main trainers
        # Parameters
        config_dict : The sub dictionary from the general training configuration specific to FLAIR
        """
        config = flairConfig()
        
        if (config != None):
            config.temp_file_sent_size = config_dict.get('temp_file_sent_size', config.temp_file_sent_size)
            config.hidden_size=config_dict.get('hidden_size', config.hidden_size)
            config.nlayers=config_dict.get('nlayers', config.nlayers)
            config.sequence_length=config_dict.get('sequence_length', config.sequence_length)
            config.mini_batch_size=config_dict.get('mini_batch_size', config.mini_batch_size)
            config.learning_rate=config_dict.get('learning_rate', config.learning_rate)
            config.epochs=config_dict.get('epochs', config.epochs)
            config.is_forward_lm=config_dict.get('is_forward_lm', config.is_forward_lm)
            config.character_level=config_dict.get('character_level', config.character_level)
            config.pretrained_model=config_dict.get('pretrained_model', config.pretrained_model)
            config.num_workers=config_dict.get('num_workers', config.num_workers)
        return config

    def get(self):
        """
        Method to return a `Dict` object from the elmoConfig for use with ray[tune]
        """
        config = {}
        config['temp_file_sent_size'] = self.temp_file_sent_size
        config['hidden_size'] = self.hidden_size
        config['nlayers'] = self.nlayers
        config['sequence_length'] = self.sequence_length
        config['mini_batch_size'] = self.mini_batch_size
        config['learning_rate'] = self.learning_rate
        config['epochs'] = self.epochs
        config['is_forward_lm'] = self.is_forward_lm
        config['character_level'] = self.character_level
        config['num_workers'] = self.num_workers
        config['pretrained_model'] = self.pretrained_model
        return config

class FLAIR_trainer(Trainer):
    """
    `Trainer` class specific to FLAIR
    """
    def __init__(self, config: Dict[str, Any] = None, 
                    logger_creator: Callable[[Dict[str, Any]], "Logger"] = None,
                    remote_checkpoint_dir: Optional[str] = None,
                    custom_syncer: Optional[Syncer] = None,):
        super().__init__(config=config, logger_creator=logger_creator,
                            remote_checkpoint_dir=remote_checkpoint_dir, 
                            custom_syncer=custom_syncer,)

    def save_checkpoint(self, checkpoint_dir):
        return
    
    def step(self):
        """
        Iteration of a ray[tune] experiment will execute this training step.
        After training, a FLAIR sequence labeller is trained to test the quality
        of the embedding
        """
        # Specify the directory where the temporary model will be trained
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)

        # Depending on the trial instance, select a GPU device to traing the FLAIR models
        cuda_instance = self.trial_number % (torch.cuda.device_count() - 1)
        print('Cuda_instance: ' + str(cuda_instance))
        if (torch.cuda.is_available()
            and
            cuda_instance > 0):
            flair.device = torch.device('cuda', cuda_instance)
        
        # Train the model
        self.__train_model__(lang_output_dir, None)
        
        # Get the training and test data for the sequence labelling task
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)

        # TODO: Specify training and test files as separate attributes in tuning parameters
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')
        # Train the sequence labeller
        acc = tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=lang_output_dir.joinpath('best-lm.pt'), 
                                    embedding_type=EmbeddingType.FLAIR, cuda_instance=cuda_instance
                                )
        # Remove the trained embedding model to save disk space for multiple runs
        shutil.rmtree(lang_output_dir)
        return {'Accuracy': acc,
                result.DONE: True}

    def name(self):
        """
        The name of the specific FLAIR embedding
        """
        if (self.config and self.config['is_forward_lm']):
            return 'FLAIR-forward'
        return "FLAIR-backward"

    def load_config_from_dict(self,  
        config_dict: Dict[str, Any] = None
    ) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object from the configuration dictionary
        specific to the FLAIR training objective
        # Parameters
        config_dict : The configuration dictionary from which to generate the TrainingConfig
        """
        return flairConfig.get_config_from_dict(config_dict)
    
    def train_local(self):
        """
        The main training iteration for FLAIR that is called either directly
        or from step() when performing hyperparameter tuning
        """
        cuda_instance = 1
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)

        if (torch.cuda.is_available()
            and
            cuda_instance > 0):
            flair.device = torch.device('cuda', cuda_instance)
        
        flair_config = flairConfig.get_config_from_dict(self.config)
        language_model = None
        # If an existing model is specified, load that model
        # and the model will be fine tuned with the new training data
        if (flair_config.pretrained_model
            and
            Path(flair_config.pretrained_model).exists()):
            try:
                language_model = FlairEmbeddings(flair_config.pretrained_model)
            except Exception:
                language_model = None
        
        # If this is not an instance where a subset of the data is used to train
        # the embedding, there is no need to generate the training data
        # as the main `Trainer` parent object already created the necessary training data 
        if (self.init_size <= 0):
            # The model output location is a combination of the name of the embedding and the specific language
            lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
            self.__train_model__(lang_output_dir, language_model)
        else:
            # Models are trained on iteratively larger data sets
            max_reached = False
            # Generate the training data for the first iteration with init_size items
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, 
                                                        flair_config.temp_file_sent_size, 
                                                    init_size=self.init_size)
            while (self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.name(), self.lang, str(self.init_size))
                self.__train_model__(lang_output_dir, language_model)
                if (max_reached):
                    return
                # Double the size of the input training data
                self.init_size *= 2
                print('Data size: '.format(str(self.init_size)))
                # Generate a new training data set
                self.input_file, max_reached = self.__process_data__(self.lang_input_dir, 
                                                flair_config.temp_file_sent_size, 
                                                init_size=self.init_size)
                                                
    def __train_model__(self, lang_output_dir: Path, language_model):
        # If the process is not fine tuning an existing model
        # generate a character dictionary and init the LanguageModel
        if not language_model:
            dictionary = self.__generate_char_dictionary__(self.input_file, lang_output_dir, 'char_dict')
            language_model = LanguageModel(dictionary,
                                           self.config['is_forward_lm'],
                                           self.config['hidden_size'],
                                           self.config['nlayers'])

        # Specify the training input as a Corpus
        corpus = TextCorpus(self.input_file,
                            language_model.dictionary,
                            language_model.is_forward_lm,
                            self.config['character_level'])
        
        # Train the model
        trainer = LanguageModelTrainer(language_model, corpus)

        # TODO: Align config to FLAIR so additional paramaters can be specified
        # config['base_path'] = lang_output_dir
        # trainer.train(**config)
        trainer.train(base_path=lang_output_dir, 
                      sequence_length=self.config['sequence_length'], 
                      mini_batch_size=self.config['mini_batch_size'], 
                      max_epochs=self.config['epochs'],
                      learning_rate=self.config['learning_rate'], 
                      checkpoint=False, 
                      num_workers=self.config['num_workers'])

    def __generate_char_dictionary__(self, file_dir: Path, output_dir: Path, output_file: Path):
        char_dictionary: Dictionary = Dictionary()
        counter = Counter()
        files = [str(x) for x in Path(file_dir).glob('**/*.*')]
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    chars = list(line)
                    counter.update(chars)
        total_count = 0
        for letter, count in counter.most_common():
            total_count += count
        sum = 0
        for letter, count in counter.most_common():
            sum += count
            percentile = (sum / total_count)

            # comment this line in to use only top X percentile of chars, otherwise filter later
            if percentile < 0.00001: break
            char_dictionary.add_item(letter)

        if (not Path(output_dir).exists()):
            Path(output_dir).mkdir(parents=True)
        with open(output_dir.joinpath(output_file), 'wb') as f:
            mappings = {
                'idx2item': char_dictionary.idx2item,
                'item2idx': char_dictionary.item2idx
            }
            pickle.dump(mappings, f)

        return char_dictionary
    
    def __process_data__(self, directories: List[Path], init_size: int = -1):
        print('Creating data sets...')
        # A simple regex to identify characters which may cause problems during training
        # and are typically from corrupted input data
        # not foolproof, but at least removes a load of problematic characters and data
        unrecognised_chars = re.compile('[^\w\s\p{P}]')

        # If there was a previous run with this training data, remove previous temp files
        tempDir = Trainer.clean_up_training_directory(directories)
        tempDir = tempDir.joinpath('corpus')
        tempDir.mkdir(parents=True)
        tempDir.joinpath("train").mkdir(parents=True)
        file_count = 1
        line_count = 0
        total_line_count = 0
        for directory in directories:
            paths = [str(x) for x in Path(directory).glob('**/*.*')]            
            with open(tempDir.joinpath('test.txt'), 'a', encoding='utf-8') as f_test, open(tempDir.joinpath('valid.txt'), 'a', encoding='utf-8') as f_validation:
                f_train = open(tempDir.joinpath('train', 'train_split_' + str(file_count)), 'a', encoding='utf-8')
                test_counter = 0
                write_validation = False
                for file in paths:
                    with open(file, 'r', encoding='utf-8') as f_read:
                        for line in f_read:
                            # Exclude empty lines and lines with corrupt/problematic characters
                            if (line.isspace()
                                or
                                unrecognised_chars.match(line)):
                                continue
                            # Write 1% of data as test and validation
                            elif test_counter == 99:
                                if write_validation:
                                    dummy = f_validation.write(line)
                                    test_counter = 0
                                    write_validation = False
                                else:
                                    dummy = f_test.write(line)
                                    write_validation = True
                                continue
                            test_counter += 1
                            line_count += 1
                            total_line_count += 1
                            dummy = f_train.write(line)
                            
                            # If we are using subsets of the data
                            # Stop when the requisite number of lines are reached
                            if (total_line_count == init_size):
                                f_train.close()
                                print(str(file_count) + ' file(s) created for training.')
                                return tempDir, False

                            # If the config specifies a max number of lines per file
                            # Create a new writer once the requisite number of lines reached
                            if (line_count == self.config['temp_file_sent_size']):
                                file_count += 1
                                line_count = 0
                                f_train.close()
                                f_train = open(tempDir.joinpath('train', 'train_split_' + str(file_count)), 'w', encoding='utf-8')
            f_train.close()

        print(str(file_count) + ' file(s) created for training.')
        return tempDir, True