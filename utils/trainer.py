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

from abc import ABC, abstractmethod
from pathlib import Path
from ray import tune
from ray.tune.syncer import Syncer
from ray.tune.schedulers import FIFOScheduler
from ray.tune.experiment import Trial
from typing import List, Dict, Any, Tuple
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ray.tune.logger import Logger

import shutil

class InternalFIFO(FIFOScheduler):
    """
    """
    trial_number = 0
    
    def on_trial_add(self, trial_runner: 'trial_runner.TrialRunner', trial: Trial):
        # Add a trial number to the config so we can determine if we can set
        # a separate GPU for this particular trial
        trial.config['trial_number'] = len(trial_runner.get_trials()) - 1

        return super().on_trial_add(trial_runner, trial)

class TrainingConfig(ABC):
    """
    """

    tuning_settings: dict = None

    def __init__(self) -> None:
        super().__init__()

    def get() -> Dict:
        pass

class Trainer(tune.Trainable):
    """

    """
    
    input_dirs: List[Path]
    tag_type = 'pos'
    lang = 'af'
    init_size = -1
    tuning_settings = {}
    output_dir: Path = None
    trial_number = 0

    def __init__(self, 
        config: Dict[str, Any] = None, 
        logger_creator: Callable[[Dict[str, Any]], "Logger"] = None,
        remote_checkpoint_dir: Optional[str] = None,
        custom_syncer: Optional[Syncer] = None
    ) -> None:
        self.lang = config.get('lang', '')
        self.tuning_settings = config.get('tuning_settings', {})
        self.init_size = config.get('init_size', -1)
        self.input_dirs = config.get('input_dirs', None)
        self.output_dir = Path(config.get('output_dir', './temp_output/'))
        self.trial_number = config.get('trial_number', 0)

        if (not self.output_dir.exists()):
            self.output_dir.mkdir(parents=True)
        
        super().__init__(self.load_config_from_dict(config).get(),
                            logger_creator=logger_creator,
                            remote_checkpoint_dir=remote_checkpoint_dir, 
                            custom_syncer=custom_syncer,)

    def setup(self, config: Dict) -> None:
        """
        """
        self.lang_input_dir = list(filter(lambda t: t.name == self.lang, 
                                            self.input_dirs))
        self.input_file, _ = self.__process_data__(self.lang_input_dir, self.init_size)
    
    @abstractmethod
    def load_config_from_dict(self, config_dict: Dict[str, Any] = None) -> TrainingConfig:
        pass

    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def train_local(self, config):
        pass

    @abstractmethod
    def __process_data__(self, input_dirs: List[Path], data_size: int) -> Tuple[Path, bool]:
        pass

    @classmethod
    def clean_up_training_directory(
        self, 
        input_dir: List[Path]
    ) -> Path:
        """
        """
        if type(input_dir) is not list: input_dir = [input_dir]
        path = [Path(x).name for x in input_dir]
        path = str.join('_', path)
        path = Path(input_dir[0]).parent.joinpath(path)
        tempDir = path.joinpath('temp')
        if (tempDir.exists()):
            shutil.rmtree(tempDir)
        return tempDir