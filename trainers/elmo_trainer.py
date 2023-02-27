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

# Code based on examples from the following github tutorials and examples
# https://github.com/PrashantRanjan09/Elmo-Tutorial/blob/master/train_elmo_updated.py
# https://github.com/allenai/bilm-tf/blob/master/bin/train_elmo.py

from allennlp.commands import train as elmo_train
from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl

from collections import Counter
from pathlib import Path
from ray.tune.syncer import Syncer
from ray.tune import result
from typing import Dict, Any, List, Union, Callable, Optional, TYPE_CHECKING
from utils.trainer import Trainer, TrainingConfig

import _jsonnet
import shutil
import regex as re

if TYPE_CHECKING:
    from ray.tune.logger import Logger

class elmoConfig(TrainingConfig):
    """
    Training configuration class for ELMO
    """
    # ELMO is somewhat unique in that it uses
    # its own jsonnet config file, rather than specifying the configuration
    # in the general training configuration
    config_file: Path = Path("./trainers/configs/bidirectional_language_model.jsonnet")

    def __init__(
            self,
            config_file: Union[str, Path] = Path("./trainers/configs/bidirectional_language_model.jsonnet")
            ):

        self.config_file = Path(config_file)

    def get_config_from_dict(config_dict: Dict[str, Any] = None) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object specifically for ELMO that can be reused by the main trainers
        # Parameters
        config_dict : The sub dictionary from the general training configuration specific to ELMO
        """
        config = elmoConfig()
        
        if (config_dict != None):
            config.config_file = Path(config_dict.get("config_file", config.config_file))

        return config

    def get(self):
        """
        Method to return a `Dict` object from the elmoConfig for use with ray[tune]
        """
        config = {}
        config['config_file'] = self.config_file
        return config

class elmo_trainer(Trainer): 
    """
    `Trainer` class specific to ELMO
    """
    def __init__(self, 
        config: Dict[str, Any] = None, 
        logger_creator: Callable[[Dict[str, Any]], "Logger"] = None,
        remote_checkpoint_dir: Optional[str] = None,
        custom_syncer: Optional[Syncer] = None
    ):
        super().__init__(config=config, logger_creator=logger_creator,
                            remote_checkpoint_dir=remote_checkpoint_dir, 
                            custom_syncer=custom_syncer,)
    
    def step(self):
        """
        Iteration of a ray[tune] experiment will execute this training step.
        After training, a FLAIR sequence labeller is trained to test the quality
        of the embedding
        """
        # Specify the directory where the temporary model will be trained
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
        # Train the model
        self.__train_model__(lang_output_dir, self.config)

        # Get the training and test data for the sequence labelling task
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')
        # Train the sequence labeller and return the score for the labelling task
        return {'Accuracy': tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=lang_output_dir, 
                                    embedding_type=EmbeddingType.ELMO
                                ),
                result.DONE: True}

    def load_config_from_dict(self, 
        config_dict: Dict[str, Any] = None
    ) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object from the configuration dictionary
        specific to the ELMO training objective
        # Parameters
        config_dict : The configuration dictionary from which to generate the TrainingConfig
        """
        return elmoConfig.get_config_from_dict(config_dict)
    
    def name(self):
        """
        The name of the specific embedding
        """
        return "ELMO"

    def train_local(self):
        """
        The main training iteration for ELMO that is called either directly
        or from step() when performing hyperparameter tuning
        """
        # If this is not an instance where a subset of the data is used to train
        # the embedding, there is no need to generate the training data
        # as the main `Trainer` parent object already created the necessary training data 
        if (self.init_size <= 0):
            # The model output location is a combination of the name of the embedding and the specific language
            lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
            self.__train_model__(lang_output_dir, self.config)
        else:
            # Models are trained on iteratively larger data sets
            max_reached = False
            # Generate the training data for the first iteration with init_size items
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, self.init_size)
            while(self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
                self.__train_model__(lang_output_dir, self.config)
                if (max_reached):
                    return
                # Double the size of the input training data
                self.init_size *= 2
                print('Data size: '.format(str(self.init_size)))
                # Generate a new training data set
                self.input_file, max_reached = self.__process_data__(self.lang_input_dir, self.init_size)
    
    def __train_model__(self, output_dir: Path, config: Dict):
        if (not output_dir.parent.exists()):
            output_dir.parent.mkdir(parents=True)

        # Get the various paths required to train ELMO
        config_dir = self.output_dir.joinpath('configs')
        validation_path = self.input_file.joinpath('validation')
        input_dir = self.input_file.joinpath('train')

        if (not config_dir.exists()):
            config_dir.mkdir(parents=True)
        # If a previous model was trained in the same location
        # remove that model
        if (output_dir.exists()):
            shutil.rmtree(output_dir)
        
        # Generate a new config with the relevant attributes
        # updated with jsonnet
        config_json = _jsonnet.evaluate_file(str(self.config['config_file']), 
                        ext_vars = { 
                            'BIDIRECTIONAL_LM_TRAIN_PATH': str(input_dir),
                            'BIDIRECTIONAL_LM_VALIDATION_PATH': str(validation_path)})
        #                   'BIDIRECTIONAL_LM_VOCAB_PATH': str(vocab_path)})
        # Write the new config (updated with jsonnet) to a temporary location
        # in order for the trainer to load the temp config file
        open(config_dir.joinpath(self.lang + '.temp.elmo.config.json'), 'w', encoding='utf8').write(config_json)

        # Train the ELMO model based on the configuration
        model = elmo_train.train_model_from_file(
                    parameter_filename=str(config_dir.joinpath(self.lang + '.temp.elmo.config.json')), 
                    serialization_dir=output_dir)

    def __process_data__(self, directories: List[Path], init_size: int=-1):
        print('Creating data sets...')
        # A simple regex to identify characters which may cause problems during training
        # and are typically from corrupted input data
        # not foolproof, but at least removes a load of problematic characters and data
        unrecognised_chars = re.compile('[^\w\s\p{P}]')

        # Remove previous training runs' data
        tempDir = self.clean_up_training_directory(directories)
        tempDir.mkdir(parents=True)
        validation_path = tempDir.joinpath('validation')
        validation_path.mkdir(parents=True)
        files = []
        for directory in directories:
            for f in directory.rglob('*.txt'):
                files.append(f)
        # Open a test and validation file for writing
        with open(tempDir.joinpath('test.txt'), 'a', encoding='utf-8') as f_test, open(validation_path.joinpath('validation.txt'), 'a', encoding='utf-8') as f_validation:
            file_count = 0
            trainDir = tempDir.joinpath('train')
            trainDir.mkdir(parents=True)
            f_train = open(trainDir.joinpath(
                  'train_split_' + str(file_count) + '.txt'), 
                  'a', encoding='utf-8')
            line_count = 0
            total_line_count = 0
            test_counter = 0
            write_validation = False
            for f in files:
                with open(f, 'r', encoding='utf8') as f_in:
                    for line in f_in:
                        # Exclude empty lines and lines with corrupt/problematic characters
                        if (line.isspace()
                            or
                            unrecognised_chars.match(line)):
                            continue
                        # Write 1% of data to the test and validation files
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
                        # Write text files in batches of 100K
                        # purely to make the training process more opaque
                        # and allow concurrent processes to have approximately
                        # the same amount of data
                        if (line_count > 100000):
                            f_train.close()
                            file_count += 1
                            f_train = open(trainDir.joinpath(
                                    'train_split_' + str(file_count) + '.txt'), 
                                    'w', encoding='utf-8')
                            line_count = 0
                        f_train.write(line)
                        line_count += 1
                        
                        # If we are using subsets of the data
                        # Stop when the requisite number of lines are reached
                        if (total_line_count == init_size):
                            f_train.close()
                            return tempDir, False
            f_train.close()
        print('Created {} files for training'.format(str(file_count)))
        return tempDir, True

