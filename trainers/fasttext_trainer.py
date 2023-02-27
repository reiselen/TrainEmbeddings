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

import fasttext
import regex as re
import torch

if TYPE_CHECKING:
    from ray.tune.logger import Logger

class fastTextConfig(TrainingConfig):
    """
    Training configuration class for fastText
    """
    model:str = 'skipgram'
    lr:float = 0.05
    dim:int = 100
    ws:int = 5
    epoch:int = 5
    minCount:int =  5
    minn:int = 3
    maxn:int = 6
    neg:int = 5
    wordNgrams:int = 1
    loss:str = 'ns'
    bucket:int = 2000000
    thread:int = 4
    lrUpdateRate:int = 100
    t:float = 0.001
    verbose:int = 2
    lang: str = 'zu'
    def __init__(
            self,
            model:str = 'skipgram',
            lr:float = 0.05,
            dim:int = 100,
            ws:int = 5,
            epoch:int = 5,
            minCount:int =  5,
            minn:int = 3,
            maxn:int = 6,
            neg:int = 5,
            wordNgrams:int = 1,
            loss:str = 'ns',
            bucket:int = 2000000,
            thread:int = 4,
            lrUpdateRate:int = 100,
            t:float = 0.001,
            verbose:int = 2):
        self.model=model
        self.lr=lr
        self.dim=dim
        self.ws=ws
        self.epoch=epoch
        self.minCount= minCount
        self.minn=minn
        self.maxn=maxn
        self.neg=neg
        self.wordNgrams=wordNgrams
        self.loss=loss
        self.bucket=bucket
        self.thread=thread
        self.lrUpdateRate=lrUpdateRate
        self.t=t
        self.verbose=verbose
    
    def get(self):
        """
        Method to return a `Dict` object from the fastTextConfig for use with ray[tune]
        """
        config = {}
        config['model'] = self.model
        config['lr'] = self.lr
        config['dim'] = self.dim
        config['ws'] = self.ws
        config['epoch'] = self.epoch
        config['minCount'] = self.minCount
        config['minn'] = self.minn
        config['maxn'] = self.maxn
        config['neg'] = self.neg
        config['wordNgrams'] = self.wordNgrams
        config['loss'] = self.loss
        config['bucket'] = self.bucket
        config['thread'] = self.thread
        config['lrUpdateRate'] = self.lrUpdateRate
        config['t'] = self.t
        config['verbose'] = self.verbose
        return config

    def get_config_from_dict(config_dict: Dict[str, Any] = None):
        """
        Generate a `TrainingConfig` object specifically for fastText that can be reused by the main trainers
        # Parameters
        config_dict : The sub dictionary from the general training configuration specific to fastText
        """
        # Get a version of the config where any ray[tune] references
        # are normalised to a single value as required for non-tuning training
        config_dict = Utils.config_to_tuning_config(config_dict)
        ft_config_to_return = fastTextConfig()
        
        if (config_dict != None):
            ft_config_to_return.model = config_dict.get('model', ft_config_to_return.model)
            ft_config_to_return.lr = config_dict.get('lr', ft_config_to_return.lr)
            ft_config_to_return.dim = config_dict.get('dim', ft_config_to_return.dim)
            ft_config_to_return.ws = config_dict.get('ws', ft_config_to_return.ws)
            ft_config_to_return.epoch = config_dict.get('epoch', ft_config_to_return.epoch)
            ft_config_to_return.minCount = config_dict.get('minCount', ft_config_to_return.minCount)
            ft_config_to_return.minn = config_dict.get('minn', ft_config_to_return.minn)
            ft_config_to_return.maxn = config_dict.get('maxn', ft_config_to_return.maxn)
            ft_config_to_return.neg = config_dict.get('neg', ft_config_to_return.neg)
            ft_config_to_return.wordNgrams = config_dict.get('wordNgrams', ft_config_to_return.wordNgrams)
            ft_config_to_return.loss = config_dict.get('loss', ft_config_to_return.loss)
            ft_config_to_return.bucket = config_dict.get('bucket', ft_config_to_return.bucket)
            ft_config_to_return.thread = config_dict.get('thread', ft_config_to_return.thread)
            ft_config_to_return.lrUpdateRate = config_dict.get('lrUpdateRate', ft_config_to_return.lrUpdateRate)
            ft_config_to_return.t = config_dict.get('t', ft_config_to_return.t)
            ft_config_to_return.verbose = config_dict.get('verbose', ft_config_to_return.verbose)
        return ft_config_to_return

class fastText_trainer(Trainer): 
    """
    `Trainer` class specific to fastText
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
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang, 
                                'fastText.' + self.lang + 
                                '.model.' + self.config['model'] +'.bin')
        # Train the model
        self.__train_model__(lang_output_dir, self.config)

        # Depending on the current trial, one of the GPUs should be selected
        # to train the FLAIR sequence labeller
        cuda_instance = self.trial_number % (torch.cuda.device_count() - 1)
        print('Cuda_instance: ' + str(cuda_instance))

        # Get the training and test data for the sequence labelling task
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')

        # Train the sequence labeller
        acc = tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=lang_output_dir, 
                                    embedding_type=EmbeddingType.FASTTEXT, cuda_instance=cuda_instance
                                )
        # Remove the trained embedding model to save disk space for multiple runs
        lang_output_dir.unlink()
        # Return the score for the labelling task
        return {'Accuracy': acc,
                result.DONE: True}

    def name(self):
        """
        The name of the specific fastText embedding
        """
        if ('model' in self.config):
            return 'fastText-' + self.config['model']
        return 'fastText'

    def load_config_from_dict(self, 
        config_dict: Dict[str, Any] = None
    ) -> TrainingConfig:
        """
        Generate a `TrainingConfig` object from the configuration dictionary
        specific to the fastText training objective
        # Parameters
        config_dict : The configuration dictionary from which to generate the TrainingConfig
        """
        return fastTextConfig.get_config_from_dict(config_dict)
    
    def train_local(self):
        """
        The main training iteration for fastText that is called either directly
        or from step() when performing hyperparameter tuning
        """
        model = self.config['model'] if 'model' in self.config else 'None'
        # If this is not an instance where a subset of the data is used to train
        # the embedding, there is no need to generate the training data
        # as the main `Trainer` parent object already created the necessary training data 
        if (self.init_size <= 0):
            # The model output location is a combination of the name of the embedding and the specific language
            lang_output_dir = self.output_dir.joinpath(self.name(), self.lang, 'fastText.' 
                                    + self.lang + '.model.' + model +'.bin')
            self.__train_model__(lang_output_dir, self.config)
        else:
            # Models are trained on iteratively larger data sets
            max_reached = False
            # Generate the training data for the first iteration with init_size items
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, self.init_size)
            while(self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.lang, 
                                                str(self.init_size), 
                                                'fastText.' + self.lang + '.model.' + model +'.bin')
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

        # Set the required input directory as part of the config
        config['input'] = str(self.input_file.resolve())
        
        model = fasttext.train_unsupervised(**config)
        model.save_model(str(output_dir))
        return

    def __process_data__(self, input_dirs: List[Path], init_size: int = -1):
        print('Creating data sets...')
        # A simple regex to identify characters which may cause problems during training
        # and are typically from corrupted input data
        # not foolproof, but at least removes a load of problematic characters and data
        unrecognised_chars = re.compile('[^\w\s\p{P}]')

        tempDir = Trainer.clean_up_training_directory(input_dirs)
        tempDir = tempDir.joinpath('corpus')
        tempDir.mkdir(parents=True)
        line_count = 0

        input_file = tempDir.joinpath('train.input.txt')
        for directory in input_dirs:
            paths = [str(x) for x in Path(directory).glob('**/*.txt')]
            with open(input_file, 'a', encoding='utf-8') as f_write:
                for file in paths:
                    with open(file, 'r', encoding='utf-8') as f_read:
                        for line in f_read:
                            # Exclude empty lines and lines with corrupt/problematic characters
                            if (line.isspace()
                                or
                                unrecognised_chars.match(line)):
                                continue
                            f_write.write(line)
                            line_count += 1
                            # If we are using subsets of the data
                            # Stop when the requisite number of lines are reached
                            if (line_count == init_size):
                                return input_file, False
        
        return input_file, True
    