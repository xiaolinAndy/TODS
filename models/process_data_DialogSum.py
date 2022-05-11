import pdb

import jsonlines
import random
import json
import os
import torch
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from tqdm import tqdm

random.seed(2021)



def process_index():
    if not os.path.exists('../data/DialogSum/aux2/'):
        os.makedirs('../data/DialogSum/aux2/')
    for name in ['train', 'val', 'test']:
        with open('../data/DialogSum/base/' + name + '.json', 'r') as f:
            data = json.load(f)
        for sample in data:
            index = sample['AllTopic'].index(sample['Topic'])
            assert len(sample['Dialogue']) == len(sample['Index'])
            for i in range(len(sample['Index'])):
                if sample['Index'][i] == index:
                    min_index = i
                    break
            for i in range(len(sample['Index']))[::-1]:
                if sample['Index'][i] == index:
                    max_index = i
                    break
            sample['AttIndexMin'] = min_index
            sample['AttIndexMax'] = max_index
            del sample['Index']
        with open('../data/DialogSum/aux2/' + name + '.json', 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

def process_contrast():
    if not os.path.exists('../data/DialogSum/aux3/'):
        os.makedirs('../data/DialogSum/aux3/')
    for name in ['train', 'val', 'test']:
        with open('aux2/' + name + '.json', 'r') as f:
            data = json.load(f)
        for sample in data:
            sample['TrueTopic'] = sample['Topic']
            other_topics = sample['AllTopic'].copy()
            other_topics.remove(sample['Topic'])
            sample['FalseTopic'] = random.choice(other_topics)
            sample['AttShannonMask'] = 1
            del sample['Topic']
        with open('../data/DialogSum/aux3/' + name + '.json', 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    if sys.argv[1] == 'aux2':
        process_index()
    elif sys.argv[1] == 'aux3':
        process_index()
        process_contrast()
