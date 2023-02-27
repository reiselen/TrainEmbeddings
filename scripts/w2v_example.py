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

# Extremely simple example script for loading word2vec models
import gensim
import numpy as np
import sys

if (len(sys.argv) < 3):
    print('Usage: python w2v_example.py <MODELFILE> <Sentence>')
    print('\t<MODELFILE>: Path to the w2v model file to load.')
    print('\t<STRINGS>: A word or sentence used to generate output.')
    exit()

try:
    model_file = sys.argv[1]
    examples = sys.argv[2:]

    print('Loading model: ' + model_file)
    f_model = gensim.models.KeyedVectors.load_word2vec_format(
                    model_file, binary=True
                )
    print('Model loaded')

    for word in examples:
        print (word)    
        if word in f_model:
            word_embedding = f_model[word]
        elif word.lower() in f_model:
            word_embedding = f_model[word.lower()]
        else:
            word_embedding = np.ones(f_model.vector_size, dtype="float")
        print(len(word_embedding))
except Exception as e:
    print(e)


