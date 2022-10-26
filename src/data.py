import json
import datasets
import os

def get_dataset(name, split):
  if name == 'webtext':
    split = 'valid' if split == 'validation' else split
    path = 'data/'
    return [json.loads(x)['text'] for x in 
        open(os.path.join(path, 'webtext.' + split +'.jsonl'))]
  elif name == 'input':
    return (input() for x in range(100))


