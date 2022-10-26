"""
Takes an mturk results file and keeps only the Answer and Input fields.
"""
import sys
import csv
import json

fields = ['Title', 'Description', 'Keywords', 'Reward', 'CreationTime', 'MaxAssignments', 'AcceptTime', 'SubmitTime', 'WorkTimeInSeconds']

worker_mapping = {}
def get_id(workerid):
  if workerid not in worker_mapping:
    worker_mapping[workerid] = len(worker_mapping)
  return worker_mapping[workerid]


input_file = sys.argv[1]

with open(input_file) as fin:
  with open('study1_results.json', 'w') as fout:
    reader = csv.DictReader(fin)
    use_fields = list(filter(lambda x: 'Answer' in x or 'Input' in x, reader.fieldnames)) + fields
    print(use_fields)
    for record in reader:
      filtered_record = {x: record[x] for x in filter(lambda k: k in use_fields, record)}
      filtered_record['OurWorkerID'] = get_id(record['WorkerId'])
      fout.write(json.dumps(filtered_record) + '\n')

print('Study 1, worker count {}'.format(len(worker_mapping)))
