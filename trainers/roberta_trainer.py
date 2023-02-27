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

# Code based on https://huggingface.co/blog/how-to-train

from evaluate.TrainTagger import EmbeddingType, TrainLabeller as tl
from pathlib import Path
from ray.tune.syncer import Syncer
from ray.tune import result
from typing import Dict, Any, List, Union, Callable, Optional, TYPE_CHECKING
from utils.trainer import Trainer, TrainingConfig

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig
from tokenizers import ByteLevelBPETokenizer

if TYPE_CHECKING:
    from ray.tune.logger import Logger

import regex as re
import shutil
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings
    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]
    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}

class roberta_training_config:
    roberta_config: RobertaConfig = None
    epochs: int = 10
    lines_per_instance: int = 5000
    model: RobertaForMaskedLM = None
    mask_percentage: float = 0.15
    batch_size: int = 64
    max_position_size: int = 512
    cuda_instance: int = 0

    def __init__(
            self,
            epochs:int = 10,
            lines_per_instances: int = 5000,
            model: Union[Path, str] = None,
            cuda_instance: int = 0,
            roberta_config = RobertaConfig(),
    ):
        self.epochs = epochs
        self.lines_per_instance = lines_per_instances
        self.roberta_config = roberta_config
        self.cuda_instance = cuda_instance
        if (model):
            self.model = RobertaForMaskedLM.from_pretrained(model)

    def get_config_from_dict(config_dict: Dict[str, Any] = None):
        config = roberta_training_config()
        if (config_dict):
            config.epochs = config_dict.get('epochs', config.epochs)
            config.model = config_dict.get('model', config.model)
            config.lines_per_instance = config_dict.get('lines_per_instance', config.lines_per_instance)
            config.mask_percentage = config_dict.get('mask_percentage', config.mask_percentage)
            config.batch_size = config_dict.get('batch_size', config.batch_size)
            config.cuda_instance = config_dict.get('cuda_instance', config.cuda_instance)
            config.max_position_size = config_dict.get('max_position_size', config.max_position_size)
            roberta_config = config_dict.get('roberta_config', config.roberta_config)
            if not(type(roberta_config) is RobertaConfig):
                config.roberta_config = RobertaConfig().from_dict(roberta_config)
             
        return config

    def get(self):
        config = {}
        config['epochs'] = self.epochs
        config['lines_per_instance'] = self.lines_per_instance
        config['roberta_config'] = self.roberta_config
        config['model'] = self.model
        config['batch_size'] = self.batch_size
        config['mask_percentage'] = self.mask_percentage
        config['max_position_size'] = self.max_position_size
        config['cuda_instance'] = self.cuda_instance
        return config

class roberta_trainer(Trainer):
    
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
        self.__train_model__(lang_output_dir, self.config)

        cuda_instance = self.config['cuda_instance']
        print('Cuda_instance: ' + str(cuda_instance))
        tuning_objective_data = self.tuning_settings.get("tuning_data_dir", None)
        train_dir = Path(tuning_objective_data).joinpath(self.lang)
        train_file = train_dir.joinpath('Data.' + self.tag_type + '.full_train.' + self.lang + '.tsv')
        test_file = train_dir.joinpath('Data.' + self.tag_type + '.test.' + self.lang + '.tsv')
        return {'Accuracy': tl().train_tagger(train_file=train_file, test_file=test_file, 
                                    embedding_path=lang_output_dir, 
                                    embedding_type=EmbeddingType.ROBERTA, cuda_instance=cuda_instance
                                ),
                result.DONE: True}

    def load_config_from_dict(self, config_dict: Dict[str, Any] = None) -> TrainingConfig:
        return roberta_training_config.get_config_from_dict(config_dict)

    def name(self):
        return "RoBERTa"

    def train_local(self):
        if (self.init_size <= 0):
            lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
            self.__train_model__(lang_output_dir, self.config)
        else:
            max_reached = False
            self.input_file, max_reached = self.__process_data__(self.lang_input_dir, self.init_size)
            while(self.input_file):
                lang_output_dir = self.output_dir.joinpath(self.name(), self.lang)
                self.__train_model__(lang_output_dir, self.config)
                if (max_reached):
                    return
                self.init_size *= 2
                print('Data size: '.format(str(self.init_size)))
                self.input_file, max_reached = self.__process_data__(self.lang_input_dir, self.init_size)

    def __train_model__(self, output_dir: Path, config: Dict):
        if (not output_dir.parent.exists()):
            output_dir.parent.mkdir(parents=True)
        dict_directory = output_dir#.joinpath('vocab')
        tokeniser = self.__trainTokeniser__(self.input_file, dict_directory, 
                            self.config['roberta_config'].vocab_size)

        if (self.config['model']):
            model = self.config['model']
        else:
            model = RobertaForMaskedLM(self.config['roberta_config'])

        cuda_instance = self.config['cuda_instance']
        print ('Cuda: ' + str(cuda_instance))
        device = torch.device('cuda', cuda_instance) if torch.cuda.is_available() else torch.device('cpu')
        ## and move our model over to the selected device
        model.to(device)
        
        model.train()
        # initialize optimizer
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
        for epoch in range(self.config['epochs']):
            paths = [str(x) for x in self.input_file.rglob('**/*.txt')]
            for path in paths:
                try:
                    print(path)
                    loader = self.__getDataLoader__(path, tokeniser)
                    # setup loop with TQDM and dataloader
                    loop = tqdm(loader, leave=True)
                    for batch in loop:
                        # initialize calculated gradients (from prev step)
                        optim.zero_grad()
                        # pull all tensor batches required for training
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)

                        # process
                        outputs = model(input_ids, attention_mask=attention_mask,
                                        labels=labels)
                        # extract loss
                        loss = outputs.loss
                        # calculate loss for every parameter that needs grad update
                        loss.backward()
                        # update parameters
                        optim.step()
                        # print relevant info to progress bar
                        loop.set_description(f'Epoch {epoch}: {path}')
                        loop.set_postfix(loss=loss.item())
                        del input_ids
                        del attention_mask
                        del labels
                except Exception as e:
                    print('Error while processing {}'.format(path))
                    print(e)
                    raise
        model.save_pretrained(str(output_dir.resolve()))
        del model
        with torch.cuda.device('cuda:' + str(cuda_instance)):
            torch.cuda.empty_cache()

    def __trainTokeniser__(self, inputDir: Path, outputDir: Path, vocab_size: int) -> RobertaTokenizer:
        if (Path(outputDir).exists()):
            shutil.rmtree(outputDir)
        Path(outputDir).mkdir(parents=True)

        paths = [str(x) for x in inputDir.rglob('**/*.txt')]
        special_tokens= ['<s>', '<pad>', '</s>', '<unk>', '<mask>']
        tokeniser = ByteLevelBPETokenizer()
        tokeniser.train(files=paths, vocab_size=vocab_size, 
                            min_frequency=3, special_tokens=special_tokens)
        tokeniser.save_model(str(outputDir.resolve()))
        print ('Tokeniser model saved to: ' + str(outputDir))
        return RobertaTokenizer.from_pretrained(str(outputDir.resolve()), 
                        max_len=self.config['max_position_size'])

    def __process_data__(self, directories: List[Path], init_size: int=-1) -> Path:
        split_size = self.config['lines_per_instance']
        unrecognised_chars = re.compile('[^\w\s\p{P}]', )

        tempDir = self.clean_up_training_directory(directories)
        tempDir.mkdir(parents=True)
        file_count = 1
        line_count = 0
        total_line_count = 0
        for directory in directories:
            paths = [str(x) for x in Path(directory).glob('**/*.*')]
            f_write = open(tempDir.joinpath('input.' + str(file_count) + '.txt'), 'w', encoding='utf-8')
            print('Processing input files for training')
            for file in paths:
                with open(file, 'r', encoding='utf-8') as f_read:
                    for line in f_read:
                        if (line.isspace()
                            or
                            unrecognised_chars.match(line)):
                            continue
                        line = line.rstrip()
                        f_write.write(line + '\n')
                        line_count += 1
                        total_line_count += 1
                        if (line_count >= split_size):
                            file_count += 1
                            line_count = 0
                            f_write.close()
                            f_write = open(tempDir.joinpath('input.' + str(file_count) + '.txt'), 'w', encoding='utf-8')
                        if (total_line_count == init_size):
                            f_write.close()
                            return tempDir, False

        f_write.close()
        print(str(file_count) + ' file(s) created for training.')
        return tempDir, True
    
    def __getDataLoader__(self, inputFile: Path, 
                            tokeniser: RobertaTokenizer) -> DataLoader:
        lines = []
        with open(inputFile, 'r', encoding='utf-8') as fp:
            lines = fp.read().split('\n')

        batch = tokeniser(lines, max_length=self.config['max_position_size'], 
                            padding='max_length', truncation=True)
        labels = torch.tensor(batch['input_ids'])
        mask = torch.tensor(batch['attention_mask'])
        # make copy of labels tensor, this will be input_ids
        input_ids = labels.detach().clone()
        # create random array of floats with equal dims to input_ids
        rand = torch.rand(input_ids.shape)
        # mask random x% where token is not 0 [PAD], 1 [CLS], or 2 [SEP]
        mask_arr = ((rand < self.config['mask_percentage']) * 
                    (input_ids != 0) * (input_ids != 1) * (input_ids != 2))
        # loop through each row in input_ids tensor (cannot do in parallel)
        for i in range(input_ids.shape[0]):
            # get indices of mask positions from mask array
            selection = torch.flatten(mask_arr[i].nonzero()).tolist()
            # mask input_ids
            input_ids[i, selection] = 4  # our custom [MASK] token == 3

        encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
        dataset = Dataset(encodings)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], 
                                shuffle=True, drop_last=True)

        return loader