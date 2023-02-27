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

from pathlib import Path

from ray import tune
from ray.air import RunConfig

from trainers.elmo_trainer import elmo_trainer
from trainers.fasttext_trainer import fastText_trainer
from trainers.flair_trainer import FLAIR_trainer
from trainers.glove_trainer import glove_trainer
from trainers.roberta_trainer import roberta_trainer
from trainers.word2vec_trainer import w2v_trainer

from utils.trainer import Trainer, InternalFIFO
from utils.utils import Utils

import torch

class EmbeddingTrainer:
    """
    Class that manages the embedding training, including hyperparameter tuning
    """
    data_size_init = -1
    use_cuda = torch.cuda.is_available()

    def __init__(self, config) -> None:
        self.config = config
        self.input_dirs = [Path(config['input_dir']).joinpath(x) for x in self.config['languages']]
        self.output_dir = Path(config['output_dir'])
        for path in self.input_dirs:
            if (not(Path(path).exists)):
                print()
                raise Exception('Could not find directory: {path}')
        
        # If value is set, the trainer will itteratively increase
        # the size of the input to create embeddings based on different
        # sizes of training data. This value is the initial size
        # from where it will be doubled on each training iteration
        if 'train_size_init' in config:
            self.data_size_init = config['train_size_init']
        else:
            self.data_size_init = -1

    def train_embeddings_from_config(self):
        """
        Main method to train different embeddings and languages based
        based on the configuration file, including hyperparameter tuning.
        """
        for alg in self.config['algorithms']:
            alg = alg.lower()
            for lang in self.config['languages']:
                lang = lang.lower()
                # If tuning_settings is set in the config, execute
                # hyperparameter tuning
                if ('tuning_settings' in self.config
                    and
                    'tuning_data_dir' in self.config['tuning_settings']):
                    tuning_config = self.config['tuning_settings']
                    # The maximum number of lines from the training data to
                    # include in the hyperparameter tuning run
                    # This can be used to specify a subset of the data
                    # but defaults to all available data
                    max_lines = tuning_config.get("max_lines", -1)
                    
                    print('Hyperparameter tuning for model: ' + alg + ' for ' + lang) 
                    max_concurrent = tuning_config.get("threads", 1)
                    tests_to_run = tuning_config.get("num", 10)
                    num_gpus = tuning_config.get('gpu', 0)
                    num_cpus = tuning_config.get('cpu', 2)
                    
                    # Create a run specific config, primarily by translating the
                    # ray[tune] options to the relevant ray[tune] types
                    run_config = self.__get_updated_config__(alg, lang, max_lines, tuning_config)

                    if (not self.use_cuda):
                        num_gpus = 0

                    tuner = tune.Tuner(
                        tune.with_resources(
                            self.__get_trainer_class__(alg),
                            resources={"gpu" : num_gpus, 
                                       "cpu": num_cpus}
                        ), 
                        param_space=run_config,
                        run_config=RunConfig(
                            name=lang + '.' + alg, 
                            verbose=1,
                        ),
                        tune_config=tune.TuneConfig(
                            mode="max", # Maximise the accuracy of the downstream task
                            num_samples=tests_to_run, 
                            max_concurrent_trials=max_concurrent,
                            metric='Accuracy',
                            scheduler=InternalFIFO()
                        )
                    )
                    # Run the tuning process
                    result = tuner.fit()
                    # Write the results of the tuning run to a language/algorithm specific file
                    df = result.get_dataframe()
                    df.to_csv('./TuningResults/{lang}.{alg}.TuningResults.csv', encoding='utf8', sep='\t')
                    # Append the best run parameters to a combined file for reference
                    with open('./TuningResults/Tuning.best_results.txt', 'a', encoding='utf8') as f_out:
                        try:
                            best_result = result.get_best_result()
                            f_out.write('{lang}\t{alg}')
                            f_out.writelines('\n')
                            f_out.write('Best result: ' + str(best_result.metrics) + str(best_result.config))
                            f_out.writelines('\n')
                        except:
                            f_out.write('{lang}\t{alg}\tN/A')
                else:
                    # Get the Trainer object that will be trained for the language
                    # and specific embedding algorithm                    
                    trainer = self.__getTrainer__(alg, lang, self.data_size_init)
                    if (not trainer):
                        continue
                    print('Training model: ' + trainer.name() + ' for ' + lang)
                    trainer.train_local()
                    trainer.clean_up_training_directory(trainer.input_dirs)
    
    def __get_updated_config__(self, algorithm_name: str, lang: str, init_size: int = -1, tuning_settings = None):
        # Check to see if this is a tuning run
        tuning = not tuning_settings == None
        # Get the updated config, with ray[tune] objects if necessary
        temp_config = Utils.config_to_tuning_config(self.config[algorithm_name], tuning)
        temp_config['lang'] = lang
        temp_config['input_dirs'] = self.input_dirs
        temp_config['output_dir'] = self.output_dir
        temp_config['init_size'] = init_size
        temp_config['tuning_settings'] = tuning_settings
        temp_config['Binary_dir'] = self.config.get('Binary_dir', './binaries') 
        return temp_config

    def __get_trainer_class__(self, reference_name):
        if reference_name == 'fasttext-s' or reference_name == 'fasttext-c':
            return fastText_trainer
        elif reference_name == "w2v-s" or reference_name == "w2v-c":
            return w2v_trainer
        elif reference_name == "flair-f" or reference_name == "flair-b":
            return FLAIR_trainer
        elif reference_name == 'glove':
            return glove_trainer
        elif reference_name == 'roberta':
            return roberta_trainer
        elif reference_name == 'elmo':
            return elmo_trainer
    
    def __getTrainer__(self, reference_name: str, lang: str, init_size: int = -1) -> Trainer:
        trainer: Trainer = None
        config = self.__get_updated_config__(reference_name, lang, init_size)
        if reference_name == 'fasttext-s' or reference_name == 'fasttext-c':
            trainer = fastText_trainer(config)
        elif reference_name == "w2v-s" or reference_name == "w2v-c":
            trainer = w2v_trainer(config)
        elif reference_name == "flair-f" or reference_name == "flair-b":
            trainer = FLAIR_trainer(config)
        elif reference_name == 'glove':
            trainer = glove_trainer(config)
        elif reference_name == 'roberta':
            trainer = roberta_trainer(config)
        elif reference_name == 'elmo':
            trainer = elmo_trainer(config)
        else:
            print('Unrecoginsed algorithm in config: ' + reference_name)
            return None
        trainer.tuning_settings = self.config.get('tuning_settings', {})
        return trainer