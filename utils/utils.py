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
from typing import Dict

import json

class Utils:
    """
    A selection of utilities that can be reused throughout the code as necessary
    """
    def load_config_from_file(self, config_file: Path) -> Dict[str, str]:
        """
        Simple json loader for a configuration file that will be used
        # Parameters

        config_file : `Path`,
            The path to the config file to load
        """
        # Verify that the config file exists
        if (not config_file.exists):
            raise Exception("Specified config file does not exist: " + str(config_file.resolve()))
        
        with open(config_file.resolve(), 'r', encoding='utf-8') as f_in:
            config = json.load(f_in)
        
        # Verify the config file (as far as possible)
        # to ensure that valid parameters are set
        error_message = self.__verify_config__(config)
        if error_message:
            print ('One or more parameters in config file are not valid.\n{error_message}')
            return None
        
        return config
    
    def config_to_tuning_config(config: Dict, tuning: bool = False) -> Dict:
        """
        Convert a config dictionary object containing string versions of the
        tuning options used by the ray[tune] module into the necessary ray[tune] types

        # Parameters
        config : `Dict`,
            The configuration dictionary object that should be processed
            and converted to the ray[tune] types where necessary
        tuning : `bool`, optional (default = False)
            If set to True will replace with the ray[tune] objects, otherwise
            only takes the lower value as default from the tuning options

        The different tuning options (keys)
        "uniform": tune.uniform(-5, -1),  # Uniform float between -5 and -1
        "quniform": tune.quniform(3.2, 5.4, 0.2),  # Round to increments of 0.2
        "loguniform": tune.loguniform(1e-4, 1e-1),  # Uniform float in log space
        "qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-5),  # Round to increments of 0.00005
        "randn": tune.randn(10, 2),  # Normal distribution with mean 10 and sd 2
        "qrandn": tune.qrandn(10, 2, 0.2),  # Round to increments of 0.2
        "randint": tune.randint(-9, 15),  # Random integer between -9 and 15
        "qrandint": tune.qrandint(-21, 12, 3),  # Round to increments of 3 (includes 12)
        "lograndint": tune.lograndint(1, 10),  # Random integer in log space
        "qlograndint": tune.qlograndint(1, 10, 2),  # Round to increments of 2
        "choice": tune.choice(["a", "b", "c"]),  # Choose one of these options uniformly
        "func": tune.sample_from(
            lambda spec: spec.config.uniform * 0.01
        ),  # Depends on other value
        "grid": tune.grid_search([32, 64, 128]),  # Search over all these values
        """
        try:
            for k, v in config.items():
                if type(v) == dict:
                    for sub_k, sub_v in v.items():
                        if sub_k.lower() == 'choice':
                            config[k] = tune.choice(sub_v) if tuning else sub_v[0]
                        elif sub_k.lower() == 'uniform':
                            config[k] = tune.uniform(sub_v[0], sub_v[1]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'quniform':
                            config[k] = tune.quniform(sub_v[0], sub_v[1], sub_v[2]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'loguniform':
                            config[k] = tune.loguniform(sub_v[0], sub_v[1]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'qloguniform':
                            config[k] = tune.qloguniform(sub_v[0], sub_v[1], sub_v[2]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'randn':
                            config[k] = tune.randn(sub_v[0], sub_v[1]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'qrandn':
                            config[k] = tune.qrandn(sub_v[0], sub_v[1], sub_v[2]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'randint':
                            config[k] = tune.randint(sub_v[0], sub_v[1]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'qrandint':
                            config[k] = tune.qrandint(sub_v[0], sub_v[1], sub_v[2]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'lograndint':
                            config[k] = tune.lograndint(sub_v[0], sub_v[1]) if tuning else sub_v[0]
                        elif sub_k.lower() == 'qlograndint':
                            config[k] = tune.qlograndint(sub_v[0], sub_v[1], sub_v[2]) if tuning else sub_v[0]
            return config
        except Exception:
            print('Invalid tuning parameters set. Please review documentation.')
            raise

    def __verify_config__(config: Dict) -> str:
        error_strings = []
        test = config.get('input_dir', None)
        if not test or not Path(test).exists():
            error_strings.append("Specified input directory does not exist.")
        
        test = config.get('algorithms', None)        
        if not test:
            error_strings.append('Required algorithms parameter not set')
        else:
            for alg in test:
                if not alg in config:
                    return error_strings.append('Specified algorithm \'{alg}\' does not define training parameters.')
        
        if (len(error_strings) > 0):
            return '\n'.join(error_strings)
        return None
