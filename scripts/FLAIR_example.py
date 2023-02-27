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

# Extremely simple example script for loading FLAIR models
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence
import sys

if (len(sys.argv) < 3):
    print('Usage: python FLAIR_example.py <MODELFILE> <Sentence>')
    print('\t<MODELFILE>: Path to the FLAIR model file to load.')
    print('\t<STRINGS>: A word or sentence used to generate output.')
    exit()

try:
    model_file = sys.argv[1]
    examples = sys.argv[2:]
    print('Loading model: ' + model_file)
    f_model = FlairEmbeddings(model_file)
    print('Model loaded')

    for ex in examples:
        print (ex)
        for s in f_model.embed(Sentence(ex)):
            for tok in s.tokens:
                print(tok)
                print(len(tok.embedding.cpu().numpy()))
except Exception as e:
    print(e)


