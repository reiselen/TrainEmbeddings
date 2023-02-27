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

# Extremely simple example script for loading RoBERTa models
import transformers
from transformers import pipeline
from transformers import RobertaModel, RobertaTokenizer
import torch

import sys

if (len(sys.argv) < 3):
    print('Usage: python RoBERTa_example.py <MODELFILE> <Sentence>')
    print('\t<MODELFILE>: Path to the directory containing the RoBERTa model to load.')
    print('\t<STRINGS>: A word or sentence used to generate output.')
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    model_file = sys.argv[1]
    examples = sys.argv[2:]

    fill_mask = pipeline("fill-mask", model=model_file, tokenizer=model_file)
    tokenizer = RobertaTokenizer.from_pretrained(model_file, output_hidden_states=True)
    model = RobertaModel.from_pretrained(model_file)
    model = model.to(device)

    for ex in examples:
        ids = tokenizer.encode(ex)
        ids = torch.LongTensor(ids)
        ids = ids.to(device)
        ids = ids.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=ids)
        print(out.last_hidden_state.shape)

    result = fill_mask(' '.join(examples) + ' <mask>')
    print(result)
except Exception as e:
    print(e)


