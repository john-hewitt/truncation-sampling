import json
from collections import defaultdict, Counter

results = [json.loads(x) for x in open('study2_results.json')]

workers = Counter([record['OurWorkerID'] for record in results])
print(workers)

stats = defaultdict(lambda: defaultdict(int))
for record in results:
  is_option_1 = record['Answer.choice.option_1'] == 'true'
  is_option_2 = record['Answer.choice.option_2'] == 'true'
  is_none = record['Answer.choice.None'] == 'true'
  is_equal = record['Answer.choice.equivalent'] == 'true'
  if int(record['WorkTimeInSeconds']) < 20:
    continue

  preferred_method = (record['Input.option_1_method'] if is_option_1
      else (record['Input.option_2_method'] if is_option_2
        else ('none' if is_none
          else 'equivalent' if is_equal else None)))
  sorted_method_names = tuple(sorted((record['Input.option_1_method'], record['Input.option_2_method'])))
  stats[sorted_method_names][preferred_method] += 1

print()
for k in stats:
  print(k, stats[k])

data = ([1 for x in range(stats[('eta_suffix', 'topp_suffix')]['eta_suffix'])]
    + [-1 for x in range(stats[('eta_suffix', 'topp_suffix')]['topp_suffix'])]
    + [0 for x in range(stats[('eta_suffix', 'topp_suffix')]['none'])]
    + [0 for x in range(stats[('eta_suffix', 'topp_suffix')]['equivalent'])]
    )

print(data)
from scipy import stats as scistat
print(scistat.wilcoxon(data))

