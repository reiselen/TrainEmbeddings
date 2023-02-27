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
from utils.utils import Utils
from embeddingTrainer import EmbeddingTrainer

import argparse
import sys
import traceback

def main():
    """
    Main entry point for training embeddings based on a config file
    """
    parser = argparse.ArgumentParser(description='Script to train embeddings for a number of different architectures given input data')
    parser.add_argument('-c', '-config', dest='config', help='Training configuration file', required=True)
    try:
        args = parser.parse_args()
        config_path = Path(args.config)
        config = Utils.load_config_from_file(config_path)
        if config:
            trainer = EmbeddingTrainer(config)
            trainer.train_embeddings_from_config()
        else:
            print('Config file could not be parsed properly for processing.')
            print('Please review the config file to get a valid config')
    except Exception:
        traceback.print_exc()
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()