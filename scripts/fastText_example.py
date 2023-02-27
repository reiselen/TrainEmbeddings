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

# Extremely simple example script for loading fastText models
import fasttext
import sys

if (len(sys.argv) < 2):
    print('Usage: python fastText_example.py <MODELFILE> <Sentence>')
    print('\t<MODELFILE>: Path to the fastText model file to load.')
    print('\t<STRINGS>: A word or sentence to use to generate output.')
    exit()

model_file = sys.argv[1]
examples = sys.argv[2:]

try:
    print('Loading model: ' + model_file)
    f_model = fasttext.load_model(model_file)
    print('Model loaded')

    for ex in examples:
        print (ex)
        nn = f_model.get_word_vector(ex)
        print(len(nn))

except Exception as e:
    print(e)


