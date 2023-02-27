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

from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl
from pathlib import Path
from ray.tune.syncer import Syncer
from ray.tune import result
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from utils.trainer import Trainer, TrainingConfig
from utils.utils import Utils

import platform
import subprocess
import regex as re
import torch

if TYPE_CHECKING:
    from ray.tune.logger import Logger

class gloveConfig(TrainingConfig):
    """
    Training configuration class for GloVe
    """
    verbose: int = 2
    memory: float = 4.0
    vocab_min_count: int = 2
    vector_size: int = 300
    max_iter: int = 50
    window_size: int = 20
    binary: int = 2
    num_threads: int = 8
    x_max: int = 10

    def __init__(
            self,
            verbose: int = 2,
            memory: float = 4.0,
            vocab_min_count: int = 2,
            vector_size: int = 300,
            max_iter: int = 50,
            window_size: int = 20,
            binary: int = 2,
            num_threads: int = 8,
            x_max: int = 10):
        self.verbose = verbose
        self.memory = memory
        self.vocab_min_count = vocab_min_count
        self.vector_size = vector_size
        self.max_iter = max_iter
        self.window_size = window_size
        self.binary = binary
        self.num_threads = num_threads
        self.x_max = x_max
    
    def get_config_from_dict(config_dict: Dict[str, Any] = None) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object specifically for GloVe that can be reused by the main trainers
        # Parameters
        config_dict : The sub dictionary from the general training configuration specific to GloVe
        """
        config_dict = Utils.config_to_tuning_config(config_dict)
        config = gloveConfig()
        
        if (config != None):
            config.verbose = config_dict.get('verbose', config.verbose)
            config.memory = config_dict.get("memory", config.memory)
            config.vocab_min_count = config_dict.get("vocab_min_count", config.vocab_min_count)
            config.vector_size = config_dict.get("vector_size", config.vector_size)
            config.max_iter = config_dict.get("max_iter", config.max_iter)
            config.window_size = config_dict.get("window_size", config.window_size)
            config.binary = config_dict.get("binary", config.binary)
            config.num_threads = config_dict.get("num_threads", config.num_threads)
            config.x_max = config_dict.get("x_max", config.x_max)
        return config
    
    def get(self):
        """
        Method to return a `Dict` object from the gloveConfig for use with ray[tune]
        """
        config = {}
        config['verbose'] = self.verbose
        config['memory'] = self.memory
        config['vocab_min_count'] = self.vocab_min_count
        config['vector_size'] = self.vector_size
        config['max_iter'] = self.max_iter
        config['window_size'] = self.window_size
        config['binary'] = self.binary
        config['num_threads'] = self.num_threads
        config['x_max'] = self.x_max
        return config

class glove_trainer(Trainer): 
    """
    `Trainer` class specific to GloVe
    """    
    BINARY_PATH: Path = Path('./').resolve()
    VOCAB_FILE = BINARY_PATH.joinpath('vocab.txt')
    COOCURRENCE_FILE = BINARY_PATH.joinpath('cooccurrence.bin')
    COOCCURRENCE_SHUF_FILE = BINARY_PATH.joinpath('cooccurrence.shuf.bin')
    VECTOR_FILE = BINARY_PATH.joinpath('vectors')

    def __init__(self, config: Dict[str, Any] = None, 
                    logger_creator: Callable[[Dict[str, Any]], "Logger"] = None,
                    remote_checkpoint_dir: Optional[str] = None,
                    custom_syncer: Optional[Syncer] = None,):
        super().__init__(config=config, logger_creator=logger_creator,
                            remote_checkpoint_dir=remote_checkpoint_dir, 
                            custom_syncer=custom_syncer,)
        self.BINARY_PATH = config.get('Binary_dir', Path('./'))

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
        
        if (not lang_output_dir.exists()):
            lang_output_dir.mkdir(parents=True)

        out_path = lang_output_dir.joinpath('vectors.txt').resolve()
        # Train the model
        self.__train_model__(lang_output_dir)

        # Depending on the current trial, one of the GPUs should be selected
        # to train the FLAIR sequence labeller
        cuda_instance = self.trial_number % (torch.cuda.device_count() - 1)

        print('Cuda_instance: ' + str(cuda_instance))
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')
        acc = tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=out_path, 
                                    embedding_type=EmbeddingType.GLOVE, cuda_instance=cuda_instance
                                )
        # Remove the trained embedding model to save disk space for multiple runs
        out_path.unlink()
        # Return the score for the labelling task
        return {'Accuracy': acc,
                result.DONE: True}

    def name(self):
        return "GloVe"
    
    def load_config_from_dict(self,
        config_dict: Dict[str, Any] = None
    ) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object from the configuration dictionary
        specific to the GloVe training objective
        # Parameters
        config_dict : The configuration dictionary from which to generate the TrainingConfig
        """
        return gloveConfig.get_config_from_dict(config_dict)
    
    def train_local(self):
        """
        The main training iteration for GloVe that is called either directly
        or from step() when performing hyperparameter tuning
        """
        # If this is not an instance where a subset of the data is used to train
        # the embedding, there is no need to generate the training data
        # as the main `Trainer` parent object already created the necessary training data 
        if (self.init_size <= 0):
            # The model output location is a combination of the name of the embedding and the specific language
            lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
            self.__train_model__(lang_output_dir)
        else:
            # Models are trained on iteratively larger data sets
            max_reached = False
            # Generate the training data for the first iteration with init_size items
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, init_size=self.init_size)
            while (self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.name(), self.lang, str(self.init_size))
                self.__train_model__(lang_output_dir)
                
                if (max_reached):
                    return
                # Double the size of the input training data
                self.init_size *= 2
                print('Data size: '.format(str(self.init_size)))
                # Generate a new training data set
                self.input_file, max_reached = self.__process_data__(self.lang_input_dir, init_size=self.init_size)
        # After completion of the training, remove temporary input files
        self.clean_up_training_directory(self.input_dirs)
    
    def __train_model__(self, lang_output_dir: Path):
        # Check that the necessary GloVe binaries exist
        if (not self.BINARY_PATH.joinpath('binaries').exists()):
            print('Required GloVe binaries not found where expected: ')
            print('\t' + str(self.BINARY_PATH.joinpath('binaries').resolve()))
            raise

        if (not lang_output_dir.exists()):
            lang_output_dir.mkdir(parents=True)
        
        glove_config = gloveConfig.get_config_from_dict(self.config)

        # Set the file locations for the files generated by GloVe and consequently used
        # by further downstream processes
        self.VOCAB_FILE = lang_output_dir.joinpath(self.VOCAB_FILE.name).resolve()
        self.COOCURRENCE_FILE = lang_output_dir.joinpath(self.COOCURRENCE_FILE.name).resolve()
        self.COOCCURRENCE_SHUF_FILE = lang_output_dir.joinpath(self.COOCCURRENCE_SHUF_FILE).resolve()
        self.VECTOR_FILE = lang_output_dir.joinpath(self.VECTOR_FILE.name).resolve()

        # Run the vocabulary builder
        proc_args = ([str(self.BINARY_PATH.joinpath('binaries', 'vocab_count').resolve()),
                      '-min-count', str(glove_config.vocab_min_count),
                      '-verbose', str(glove_config.verbose)])
        self.__run_glove_binaries__(proc_args, self.input_file,
                                    self.VOCAB_FILE, False)

        # Run the cooccurrence binary
        proc_args = ([str(self.BINARY_PATH.joinpath('binaries', 'cooccur').resolve()),
                      '-memory', str(glove_config.memory),
                      '-vocab-file', str(self.VOCAB_FILE), 
                      '-verbose', str(glove_config.verbose),
                      '-window-size', str(glove_config.window_size)])
        self.__run_glove_binaries__(proc_args, self.input_file, 
                            self.COOCURRENCE_FILE, False)

        # Run the shuffle binary
        proc_args = ([str(self.BINARY_PATH.joinpath('binaries', 'shuffle').resolve()),
                       '-memory', str(glove_config.memory),
                       '-verbose', str(glove_config.verbose)])
        self.__run_glove_binaries__(proc_args, str(self.COOCURRENCE_FILE), 
                                    str(self.COOCCURRENCE_SHUF_FILE), False)

        # Run the final step of creating the embeddings with the glove binary
        log_file_name = 'log.' + '.'.join([str(glove_config.vector_size), 
                        str(glove_config.max_iter), 
                        str(glove_config.window_size)]) + '.txt'
        proc_args = ([str(self.BINARY_PATH.joinpath('binaries', 'glove').resolve()),
                      '-save-file', str(self.VECTOR_FILE), 
                      '-threads', str(glove_config.num_threads), 
                      '-input-file', str(self.COOCCURRENCE_SHUF_FILE), 
                      '-x-max', str(glove_config.x_max), 
                      '-iter', str(glove_config.max_iter), 
                      '-vector-size', str(glove_config.vector_size), 
                      '-binary', str(glove_config.binary), 
                      '-vocab-file', str(self.VOCAB_FILE), 
                      '-verbose', str(glove_config.verbose), 
                      '-write-header', '1', 
                      '>>', str(lang_output_dir.joinpath(log_file_name))])
        self.__run_glove_binaries__(proc_args, None, 
                            log_file_name, glove_config.binary != 2)
        return

    def __process_data__(self, directories: List[Path], init_size: int = -1):
        print('Creating data sets...')
        # A simple regex to identify characters which may cause problems during training
        # and are typically from corrupted input data
        # not foolproof, but at least removes a load of problematic characters and data
        unrecognised_chars = re.compile('[^\w\s\p{P}]')
        tempDir = self.clean_up_training_directory(directories)
        tempDir.mkdir(parents=True)
        total_line_count = 0

        for directory in directories:
            paths = [str(x) for x in directory.glob('**/*.*')]
            print(paths)
            tempDir = tempDir.joinpath('train.input.txt')
            with open(tempDir, 'a', encoding='utf-8') as f_write:
                for file in paths:
                    lineCount = 0
                    with open(file, 'r', encoding='utf-8') as f_read:
                        for line in f_read:
                            # Exclude empty lines and lines with corrupt/problematic characters
                            if (line.isspace()
                                or
                                unrecognised_chars.match(line)):
                                continue
                            if (lineCount < 30):
                                f_write.write(line.rstrip('\n'))
                                f_write.write(' ')
                                lineCount += 1
                            else:
                                lineCount = 0
                                f_write.write(line)
                            total_line_count += 1
                            if (total_line_count == init_size):
                                print('Created data file for Glove training.')
                                return tempDir, False

        print('Created data file for Glove training.')
        return tempDir, True
    
    def __run_glove_binaries__(self, args: List[str], std_in: Path, 
                                std_out: Path):
        """
        Simple wrapper class for executing Glove binaries with the specified commanline
        arguments
        """
        if platform.system() == 'Windows':
            args[0] = args[0] + '.exe'
        write_type = 'w' if self.config['binary'] == 2 else 'wb'
        with open(std_out, write_type, encoding='utf8') as f_out:
            if (std_in):
                with open(std_in, 'r', encoding='utf8') as f_in:
                    proc_result = subprocess.run(args, stdin=f_in, stdout=f_out)
            else:
                proc_result = subprocess.run(args, stdout=f_out)

        print(proc_result)
