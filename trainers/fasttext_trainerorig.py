#https://docs.ray.io/en/latest/tune/index.html

from cgi import test
from utils.trainer import Trainer, TrainingConfig
from pathlib import Path
from typing import Dict, Any, List
from utils.utils import Utils
import numpy as np
import fasttext

from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl

class fastTextConfig(TrainingConfig):
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
        config_dict = Utils.config_to_tuning_config(config_dict)
        config = fastTextConfig()
        
        if (config_dict != None):
            config.model = config_dict.get('model', config.model)
            config.lr = config_dict.get('lr', config.lr)
            config.dim = config_dict.get('dim', config.dim)
            config.ws = config_dict.get('ws', config.ws)
            config.epoch = config_dict.get('epoch', config.epoch)
            config.minCount = config_dict.get('minCount', config.minCount)
            config.minn = config_dict.get('minn', config.minn)
            config.maxn = config_dict.get('maxn', config.maxn)
            config.neg = config_dict.get('neg', config.neg)
            config.wordNgrams = config_dict.get('wordNgrams', config.wordNgrams)
            config.loss = config_dict.get('loss', config.loss)
            config.bucket = config_dict.get('bucket', config.bucket)
            config.thread = config_dict.get('thread', config.thread)
            config.lrUpdateRate = config_dict.get('lrUpdateRate', config.lrUpdateRate)
            config.t = config_dict.get('t', config.t)
            config.verbose = config_dict.get('verbose', config.verbose)
        return config

class fastText_trainer(Trainer):
    tag_type = 'pos'
    lang = 'af'
    init_size = -1

    def __init__(self, input_dirs: List[Path], output_dir: Path, config_dict: Dict[str, Any] = None) -> Trainer:
        super().__init__(input_dirs, output_dir, config_dict)

    def name(self):
        if (self.config and self.config.model):
            return 'fastText-' + self.config.model
        return 'fastText'

    def set_language(self, lang: str):
        self.lang = lang
        self.lang_input_dir = list(filter(lambda t: t.name == self.lang, self.input_dirs))
        self.input_file, _ = self.___process_data___(self.lang_input_dir)
    
    def set_data_size_init(self, init_size=-1):
        self.init_size = init_size

    def load_config_from_dict(self, config_dict: Dict[str, Any] = None):
        return fastTextConfig.get_config_from_dict(config_dict)
    
    def train(self, config):
        if (self.init_size <= 0):
            lang_output_dir = self.output_dir.joinpath(self.lang, 
                                    str(np.random.randint(0,100)) + '.fastText.' 
                                    + self.lang + '.model.' + self.config.model +'.bin')
            self.__train_model__(lang_output_dir, config)
        else:
            max_reached = False
            self.input_file, max_reached = self.___process_data___(self.lang_input_dir, self.init_size)
            while(self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.lang, str(self.init_size), 'fastText.' + self.lang + '.model.' + self.config.model +'.bin')
                self.__train_model__(lang_output_dir, config)
                if (max_reached):
                    return
                self.init_size *= 2
                print('Data size: '.format(str(self.init_size)))
                self.input_file, max_reached = self.___process_data___(self.lang_input_dir, self.init_size)
        
        if (self.config.tuning_train_dir == None):
            return 1.0
        else:
            train_dir = Path(self.config.tuning_train_dir).joinpath(self.lang)
            train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
            test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')
            return tl().train_tagger(train_file=train_file, test_file=test_file, 
                                        embedding_path=lang_output_dir, embedding_type=EmbeddingType.FASTTEXT)

    def __train_model__(self, output_dir: Path, config: Dict):
        if (not output_dir.parent.exists()):
            output_dir.parent.mkdir(parents=True)

        config['input'] = str(self.input_file.resolve())
        model = fasttext.train_unsupervised(**config)
                    #model=self.config.model,
                    #lr=self.config.lr,
                    #dim=self.config.dim,
                    #ws=self.config.ws,
                    #epoch=self.config.epoch,
                    #minCount=self.config.minCount,
                    #minn=self.config.minn,
                    #maxn=self.config.maxn,
                    #neg=self.config.neg,
                    #wordNgrams=self.config.wordNgrams,
                    #loss=self.config.loss,
                    #bucket=self.config.bucket,
                    #thread=self.config.thread,
                    #lrUpdateRate=self.config.lrUpdateRate,
                    #t=self.config.t,
                    #verbose=self.config.verbose)
        model.save_model(str(output_dir))
        #Trainer.clean_up_training_directory(self.input_dirs)
        return

    def ___process_data___(self, input_dirs: List[Path], data_size: int = -1):
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
                            f_write.write(line)
                            line_count += 1
                            if (data_size > 0 and line_count >= data_size):
                                return input_file, False
        if (data_size > line_count):
            return input_file, True
        
        return input_file, True
    