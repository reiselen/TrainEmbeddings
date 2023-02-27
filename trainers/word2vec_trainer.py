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

from utils.trainer import Trainer, TrainingConfig
from utils.utils import Utils
from pathlib import Path
from typing import Dict, Any, List
from gensim.models import Word2Vec
import torch
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from ray.tune.syncer import Syncer
from ray.tune import result
if TYPE_CHECKING:
    from ray.tune.logger import Logger

from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl


class word2VecConfig (TrainingConfig):
    vector_size: int=100
    window: int=5
    min_count: int=2
    workers: int=1
    epochs: int=10
    sg: int=1 # Algorithm: 0=CBOW, 1=Skipgram
    hs: int=1 # hierarchical softmax, 0 = negative sampling, if negative set
    negative: int=0 # set greater than 1 to use negative sampling
    ns_exponent: float =0.75
    cbow_mean: int=0 # sum of context word vectors, 1= use mean
    alpha: float =1 # initial learning rate
    min_alpha:float  = 0.0001 # learning rate drops linearly to min_alpha
    seed: int=None
    max_vocab_size: int =None
    max_final_vocab: int =None
    sample: float =0
    compute_loss: bool =False
    
    def __init__(
            self,
            vector_size: int =100,
            window: int =5,
            min_count: int =2,
            workers: int =1,
            epochs: int =5,
            sg: int =1, # Algorithm: 0=CBOW, 1=Skipgram
            hs: int =1, # hierarchical softmax, 0 = negative sampling, if negative set
            negative: int =0, # set greater than 1 to use negative sampling
            ns_exponent: float =0.75,
            cbow_mean: int =0, # sum of context word vectors, 1= use mean
            alpha: float =1, # initial learning rate
            min_alpha:float = 0.0001, # learning rate drops linearly to min_alpha
            seed: int=None,
            max_vocab_size: int=None,
            max_final_vocab: int =None,
            sample: float=0,
            compute_loss: bool=False):
        self.vector_size=int(vector_size)
        self.window=int(window)
        self.min_count=int(min_count)
        self.workers=int(workers)
        self.epochs=int(epochs)
        self.sg = int(sg)
        self.hs=int(hs)
        self.negative=int(negative)
        self.ns_exponent=float(ns_exponent)
        self.cbow_mean=int(cbow_mean) # sum of context word vectors, 1= use mean
        self.alpha=float(alpha) # initial learning rate
        self.min_alpha = float(min_alpha) # learning rate drops linearly to min_alpha
        self.seed= int(seed) if seed else None
        self.max_vocab_size= int(max_vocab_size) if max_vocab_size else None
        self.max_final_vocab = int(max_final_vocab) if max_final_vocab else None
        self.sample=float(sample) if sample else None
        self.compute_loss=bool(compute_loss)
    
    def get_config_from_dict(config_dict: Dict[str, Any] = None):
        config_dict = Utils.config_to_tuning_config(config_dict)
        config = word2VecConfig()
        
        if (config != None):
            config.window = config_dict.get('window',config.window)
            config.min_count = config_dict.get('min_count', config.min_count)
            config.workers = config_dict.get('workers',config.workers)
            config.vector_size = config_dict.get('vector_size',config.vector_size)
            config.sg = config_dict.get('sg', config.sg)
            config.hs = config_dict.get('hs', config.hs)
            config.epochs = config_dict.get('epochs', config.epochs)
            config.negative = config_dict.get('negative', config.negative)
            config.ns_exponent = config_dict.get('ns_exponent', config.ns_exponent)
            config.cbow_mean = config_dict.get('cbow_mean', config.cbow_mean)
            config.alpha = config_dict.get('alpha', config.alpha)
            config.min_alpha = config_dict.get('min_alpha',config.min_alpha)
            config.seed = config_dict.get('seed', config.seed)
            config.max_vocab_size = config_dict.get('max_vocab_size', config.max_vocab_size)
            config.max_final_vocab = config_dict.get('max_final_vocab', config.max_final_vocab)
            config.sample = config_dict.get('sample', config.sample)
            config.compute_loss = config_dict.get('compute_loss', config.compute_loss)
        return config

    def get(self):
        config = {}
        config['window'] = self.window
        config['min_count'] = self.min_count
        config['workers'] = self.workers
        config['vector_size'] = self.vector_size
        config['sg'] = self.sg
        config['hs'] = self.hs
        config['epochs'] = self.epochs
        config['negative'] = self.negative
        config['ns_exponent'] = self.ns_exponent
        config['cbow_mean'] = self.cbow_mean
        config['alpha'] = self.alpha
        config['min_alpha'] = self.min_alpha
        config['seed'] = self.seed
        config['max_vocab_size'] = self.max_vocab_size
        config['max_final_vocab'] = self.max_final_vocab
        config['sample'] = self.sample
        config['compute_loss'] = self.compute_loss

        return config
    
class w2v_trainer(Trainer):

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
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
        if (not lang_output_dir.exists()):
            lang_output_dir.mkdir(parents=True)
        
        type_val = self.config.get('sg', 0)
        type_val = 'skipgram' if type_val == 1 else 'cbow'
        out_path = lang_output_dir.joinpath('w2v.' + self.lang + '.model.' + type_val +'.bin').resolve()

        print(out_path)
        self.__train_model__(out_path)

        cuda_instance = self.trial_number % (torch.cuda.device_count() - 1)

        print('Cuda_instance: ' + str(cuda_instance))
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')

        acc = tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=out_path, 
                                    embedding_type=EmbeddingType.W2V, cuda_instance=cuda_instance
                                )
        out_path.unlink()
        return {'Accuracy': acc,
                result.DONE: True}
    
    def name(self):
        if (self.config.get('sg', 0) == 1):
            return 'w2v-skipgram'
        return 'w2v-cbow'

    def load_config_from_dict(self, config_dict: Dict[str, Any] = None) -> TrainingConfig:
        return word2VecConfig.get_config_from_dict(config_dict)

    def train_local(self):
        lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)

        if (not lang_output_dir.exists()):
            lang_output_dir.mkdir(parents=True)
        type = 'skipgram' if self.config.get('sg', 0) == 1 else 'cbow'

        data_size_init = self.init_size
        if (data_size_init <=0):
            out_path = lang_output_dir.joinpath('w2v.' + self.lang + '.model.' + type +'.bin').resolve()
            self.__train_model__(out_path)
        else:
            max_reached = False
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, data_size=data_size_init)
            while (self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.lang, str(data_size_init))
                out_path = lang_output_dir.joinpath('w2v.' + self.lang + '.model.' + type +'.bin'
                                                    ).resolve()
                self.__train_model__(out_path)

                if (max_reached):
                    return
                data_size_init *= 2
                print('Data size: '.format(str(data_size_init)))
                self.input_file, max_reached = self.__process_data__(self.lang_input_dir, data_size=data_size_init)
                

        Trainer.clean_up_training_directory(self.input_dirs)

    def __train_model__(self, output_path):
        training_iterator = SentenceIterator(self.input_file)
        model = Word2Vec(**self.config)
        model.build_vocab(training_iterator, progress_per=10000)
        try:
            model.train(corpus_iterable=training_iterator, total_examples=model.corpus_count, epochs=model.epochs, report_delay=1)
            model.wv.save_word2vec_format(str(output_path), binary=True)
        except Exception as e:
            print(e)

    def __process_data__(self, input_dirs: List[Path], data_size: int = -1):
        print('Creating data sets...')
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
                            if (line.isspace()):
                                continue
                            line_count += 1
                            f_write.write(line)
                            
                            if (data_size > 0 and line_count >= data_size):
                                return input_file, False

        return input_file, True

## Code from Anna Krogager: https://stackoverflow.com/questions/55086734/train-gensim-word2vec-using-large-txt-file
class SentenceIterator:
    def __init__(self, filepath: Path) -> None:
        self.inputPath = filepath
    
    def __iter__(self):
        with open(self.inputPath, 'r', encoding='utf8') as f_in:
            for line in f_in:
                yield line.strip().split(' ')