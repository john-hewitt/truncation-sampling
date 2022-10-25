import sys
import json
import csv
import re
import copy
import random

random.seed(1)

fieldnames = ['uuid', 'option_1', 'option_2', 'option_1_method', 'option_2_method']
json_fieldnames = ['eta', 'topp', 'human', 'eta_suffix', 'topp_suffix', 'gold_suffix', 'prefix']
fieldnames = fieldnames + json_fieldnames

def remove_emoji(string):
    """
    From https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b user slowkow
    """
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

with open(sys.argv[1] + '.csv', 'w') as fout:
  written_header = False
  writer = csv.DictWriter(fout, fieldnames=fieldnames)
  writer.writeheader()
  uuid = 0
  lines = [x for x in open(sys.argv[1])]
  rows = []
  for line in lines[:100]:
    line = json.loads(line)
    for pair in (('eta_suffix', 'topp_suffix'), ('eta_suffix', 'gold_suffix'), ('topp_suffix', 'gold_suffix')):
      prefix = line['prefix']
      suffix_0 = line[pair[0]]
      suffix_1 = line[pair[1]]
      choice = random.randint(0,1)
      option_1, option_1_method = [(suffix_0, pair[0]), (suffix_1, pair[1])][choice]
      option_2, option_2_method = [(suffix_0, pair[0]), (suffix_1, pair[1])][1-choice]
      line['option_1'] = option_1
      line['option_2'] = option_2
      line['option_1_method'] = option_1_method
      line['option_2_method'] = option_2_method
      line['uuid'] = str(uuid)
      uuid += 1
      rows.append({k: remove_emoji(line[k]) for k in line})
      #writer.writerow(line)
  random.shuffle(rows)
  for row in rows:
    writer.writerow(row)

