"""
Generates texts from two different sampling methods,
wherein the prefix from one method is given to both
at a position where the two methods strongly differ
in their probability estimates, and generates
continuations from both models.
"""

import argparse
import warpers
import urllib
import json
import random

import transformers
import torch
from tqdm import tqdm
import data
import math
import matplotlib.pyplot as plt

argp = argparse.ArgumentParser()
argp.add_argument('--sample_count', type=int, default = 10)
argp.add_argument('--generated', type=bool, default = False)
argp.add_argument('--dataset_name', default = 'webtext')
argp.add_argument('--max_length', type=int, default = 512)
argp.add_argument('--prefix_length', type=int, default=1)
argp.add_argument('--model_string', default = 'gpt2-xl')
argp.add_argument('--cache_dir', default = None)
argp.add_argument('--device', default = 'cuda')
argp.add_argument('--seed', default=-1, type=int)
argp.add_argument('--name', default='out')
args = argp.parse_args()

if args.seed >= 0:
  torch.manual_seed(args.seed)
  random.seed(args.seed)

try:
  model = transformers.GPT2LMHeadModel.from_pretrained(args.model_string, cache_dir=args.cache_dir).to(args.device)
  tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_string, cache_dir=args.cache_dir)
except urllib.error.URLError:
  model = transformers.GPT2LMHeadModel.from_pretrained(args.model_string, cache_dir=args.cache_dir, local_files_only=True).to(args.device)
  tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_string, cache_dir=args.cache_dir, local_files_only=True)
model.config.pad_token_id = model.config.eos_token_id

model.eval()
for p in model.parameters():
  p.requires_grad = False


epsilon_range = list(sorted((0.4, 0.3, 0.2,  0.1, 0.01, 0.002,0.004, 0.008, 0.003,  0.001, 0.0006, 0.0003, 0.0009, 0.0001, 0.00005, 3e-5,0.00001, 0.000001, 0.00000001, 0.00000000001)))
h_epsilon_range = epsilon_range
h_epsilon_range = list(sorted((12.8, 6.4, 3.2, 0.8, 0.4, 0.3, 0.2,  0.1, 0.01, 0.002,0.004, 0.008, 0.003,  0.001, 0.0006, 0.0003, 0.0009, 0.0001, 0.00005, 3e-5,0.00001, 0.000001, 0.00000001, 0.00000000001)))
p_range = (.9999, .999, .99, .98, .95, .92, .9, .85, .8, .7, .6, .5, .4, .3, .2, .1)

def jensen_shannon(p, q):
  p = p.squeeze()
  q = q.squeeze()
  m = 0.5 * (p+q)
  return 0.5*(torch.nn.functional.kl_div(m.log(), p, reduction='batchmean') + torch.nn.functional.kl_div(m.log(), q, reduction='batchmean'))

def get_stats_of_sequence(warper, model_outputs=None, input_ids=None, start_index=0, oracle_model=None):
  if model_outputs is None:
    model_outputs = model(input_ids=input_ids)
  logits = model_outputs.logits
  if oracle_model:
    oracle_outputs = model(input_ids=input_ids)
  tv_h_data = []
  for index in range(logits.size()[1]):
    if index < start_index:
      continue
    last_word_logits = logits[:,index-1,:]
    unwarped_probs = torch.softmax(last_word_logits, dim=-1)
    warped_scores = warper(None, last_word_logits)
    warped_probs = torch.softmax(warped_scores, dim=-1)
    orig_entropy = torch.distributions.Categorical(probs=unwarped_probs).entropy()
    warped_entropy = torch.distributions.Categorical(probs=warped_probs).entropy()
    tv = 1 - torch.sum(torch.where(warped_scores.isinf(), torch.zeros_like(unwarped_probs), unwarped_probs))
    js = jensen_shannon(warped_probs, unwarped_probs)
    tv_h_data.append([tv.item(), orig_entropy.item(), warped_entropy.item()])
  return tv_h_data

def generate_sequence(input_ids, warper, new_length):
  """
  Arguments:
    input_ids: Tensor of size (1, len)
  Returns:
    torch.Tensor of size (1, input_ids.size()[-1] + args.infix_length)
  """
  logits_processor = transformers.LogitsProcessorList(
      [
          transformers.MinLengthLogitsProcessor(input_ids.size()[-1]+new_length, eos_token_id=model.config.eos_token_id),
      ]
  )
  outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=warper,
      output_scores=True, 
      return_dict_in_generate=True,
      stopping_criteria=transformers.StoppingCriteriaList([transformers.MaxLengthCriteria(max_length=input_ids.size()[-1]+new_length)]))
  return outputs


epsilon_warpers = {x:warpers.EpsilonWarper(x) for x in epsilon_range}
hepsilon_warpers = {x:warpers.EtaWarper(x) for x in epsilon_range}
p_warpers = {x:warpers.TopPLogitsWarper(x) for x in p_range}
typ_warpers = {x:warpers.TypicalLogitsWarper(x) for x in p_range}

# Get stats on naturalistic data
gold_epsilon_stats = []
gold_hepsilon_stats = []
gold_p_stats = []
gold_typ_stats = []
for string in tqdm(data.get_dataset(args.dataset_name, 'valid')[:args.sample_count]):
  gold_input_ids = tokenizer('<|endoftext|>'+string, return_tensors='pt').input_ids.to(args.device)[:,:args.max_length]
  model_outputs = model(input_ids=gold_input_ids)
  for epsilon in epsilon_range:
    warper = epsilon_warpers[epsilon]
    if args.generated and False:
      generated_ids = generate_sequence(gold_input_ids[:,:args.prefix_length], warper, args.max_length - args.prefix_length).sequences
      model_outputs = model(input_ids=generated_ids)
      gold_epsilon_stats.extend([[epsilon] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs)])
    else:
      gold_epsilon_stats.extend([[epsilon] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs, start_index=args.prefix_length)])
    warper = hepsilon_warpers[epsilon]
    gold_hepsilon_stats.extend([[epsilon] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs, start_index=args.prefix_length)])
  for p in p_range:
    warper = p_warpers[p]
    if args.generated and False:
      generated_ids = generate_sequence(gold_input_ids[:,:args.prefix_length], warper, args.max_length - args.prefix_length).sequences
      model_outputs = model(input_ids=generated_ids)
      gold_p_stats.extend([[p] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs, start_index=args.prefix_length)])
    else:
      gold_p_stats.extend([[p] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs, start_index=args.prefix_length)])
    warper = typ_warpers[p]
    gold_typ_stats.extend([[p] + x for x in get_stats_of_sequence(warper, model_outputs=model_outputs, start_index=args.prefix_length)])

plt.scatter(x=[x[1] for x in gold_epsilon_stats], y = [x[3]/x[2] for x in gold_epsilon_stats], c='blue', alpha=0.1, label='e')
plt.scatter(x=[x[1] for x in gold_p_stats], y = [x[3]/x[2] for x in gold_p_stats], c='red', alpha=0.1, label='p')
plt.legend()
plt.xlabel('TV(Pre,Post)')
plt.ylabel('H(Post)/H(Pre)')
plt.title(args.dataset_name)
plt.savefig(args.name + 'plt-ratios.png')
plt.clf()
plt.scatter(x=[x[1] for x in gold_epsilon_stats], y = [x[3] for x in gold_epsilon_stats], c='blue', alpha=0.1, label='e')
plt.scatter(x=[x[1] for x in gold_p_stats], y = [x[3] for x in gold_p_stats], c='red', alpha=0.1, label='p')
plt.xlabel('TV(Pre,Post)')
plt.ylabel('H(post)')
plt.title(args.dataset_name)
plt.legend()
plt.savefig(args.name + 'plt-abs.png')
plt.clf()
plt.scatter(x=[x[2] for x in gold_epsilon_stats], y = [x[3] for x in gold_epsilon_stats], c='blue', alpha=0.1, label='e')
plt.scatter(x=[x[2] for x in gold_p_stats], y = [x[3] for x in gold_p_stats], c='red', alpha=0.1, label='p')
plt.xlabel('H(pre)')
plt.ylabel('H(post)')
plt.title(args.dataset_name)
plt.legend()
plt.savefig(args.name + 'plt-pre-post.png')
plt.clf()

fsize = 17
for h_range in ((.01,.25), (.25, .5), (.5, 1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8),(0,100)):
  print(h_range)
  epsilon_averages = []
  for e in epsilon_range:
    elts = list(filter(lambda x: x[0] == e and x[2] < h_range[1] and x[2] > h_range[0], gold_epsilon_stats))
    print('count', len(elts))
    quantiles = torch.quantile(torch.tensor([x[3] for x in elts]), q=torch.tensor([.3,.7]))
    twentieth_quantile = quantiles[0].item()
    eightieth_quantile = quantiles[1].item()
    all_elts = list(filter(lambda x: x[0] == e, gold_epsilon_stats))
    avg_tv = sum([x[1] for x in all_elts])/len(all_elts)
    avg_h = sum([x[3] for x in elts])/len(elts)
    avg_h_ratio = sum([x[3]/x[2] for x in elts])/len(elts)
    print(e, avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile)
    epsilon_averages.append((avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile))

  hepsilon_averages = []
  for e in epsilon_range:
    elts = list(filter(lambda x: x[0] == e and x[2] < h_range[1] and x[2] > h_range[0], gold_hepsilon_stats))
    quantiles = torch.quantile(torch.tensor([x[3] for x in elts]), q=torch.tensor([.3,.7]))
    twentieth_quantile = quantiles[0].item()
    eightieth_quantile = quantiles[1].item()
    all_elts = list(filter(lambda x: x[0] == e, gold_hepsilon_stats))
    avg_tv = sum([x[1] for x in all_elts])/len(all_elts)
    avg_h = sum([x[3] for x in elts])/len(elts)
    avg_h_ratio = sum([x[3]/x[2] for x in elts])/len(elts)
    print(e, avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile)
    hepsilon_averages.append((avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile))

  p_averages = []
  print()
  for p in p_range:
    elts = list(filter(lambda x: x[0] == p and  x[2] < h_range[1] and x[2] > h_range[0], gold_p_stats))
    quantiles = torch.quantile(torch.tensor([x[3] for x in elts]), q=torch.tensor([.3,.7]))
    twentieth_quantile = quantiles[0].item()
    eightieth_quantile = quantiles[1].item()
    all_elts = list(filter(lambda x: x[0] == p, gold_p_stats))
    avg_tv = sum([x[1] for x in all_elts])/len(all_elts)
    avg_h = sum([x[3] for x in elts])/len(elts)
    avg_h_ratio = sum([x[3]/x[2] for x in elts])/len(elts)
    print(p, avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile)
    p_averages.append((avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile))

  typ_averages = []
  print()
  for p in p_range:
    elts = list(filter(lambda x: x[0] == p and  x[2] < h_range[1] and x[2] > h_range[0], gold_typ_stats))
    quantiles = torch.quantile(torch.tensor([x[3] for x in elts]), q=torch.tensor([.3,.7]))
    twentieth_quantile = quantiles[0].item()
    eightieth_quantile = quantiles[1].item()
    all_elts = list(filter(lambda x: x[0] == p, gold_typ_stats))
    avg_tv = sum([x[1] for x in all_elts])/len(all_elts)
    avg_h = sum([x[3] for x in elts])/len(elts)
    avg_h_ratio = sum([x[3]/x[2] for x in elts])/len(elts)
    print(p, avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile)
    typ_averages.append((avg_tv, avg_h, avg_h_ratio, twentieth_quantile, eightieth_quantile))


  # Quantiles
  plt.scatter(x=[x[0] for x in epsilon_averages], y=[x[2] for x in epsilon_averages], c='blue', alpha=0.5, label='e-2', marker='o')
  plt.scatter(x=[x[0] for x in hepsilon_averages], y=[x[2] for x in hepsilon_averages], c='green', alpha=0.5, label='e-2', marker='s')
  plt.scatter(x=[x[0] for x in p_averages], y=[x[2] for x in p_averages], c='red', alpha=0.5, label='p-2', marker='o')
  plt.scatter(x=[x[0] for x in p_averages], y=[x[4] for x in p_averages], c='red', alpha=0.5, label='p-8', marker='x')
  plt.xlabel('TV(Pre,Post)')
  plt.ylabel('H(post)')
  plt.title(args.dataset_name + ' TV' + ' H(pre) in ' + str(h_range))
  plt.legend()
  plt.savefig(args.name + 'plt-quantile-{}abs.png'.format(h_range))
  plt.clf()

  # Average means
  plt.scatter(x=[x[0] for x in epsilon_averages], y=[x[1] for x in epsilon_averages], c='blue', alpha=0.7, label='epsilon')
  plt.scatter(x=[x[0] for x in hepsilon_averages], y=[x[1] for x in hepsilon_averages], c='green', alpha=0.7, label='eta')
  plt.scatter(x=[x[0] for x in p_averages], y=[x[1] for x in p_averages], c='red', alpha=0.7, label='top-p')
  plt.scatter(x=[x[0] for x in typ_averages], y=[x[1] for x in typ_averages], c='purple', alpha=0.7, label='typical')
  plt.xlabel('Average Total Variation', fontsize=fsize)
  plt.ylabel('Post-Truncation Entropy', fontsize=fsize)
  plt.title('{} to {}-Bit Entropy Tradeoff'.format(h_range[0], h_range[1]), fontsize=fsize+3)# + ' JS' + ' H(pre) in ' + str(h_range))
  plt.legend(fontsize=fsize)
  plt.savefig(args.name + 'plt-avg-{}abs.png'.format(h_range))

  plt.clf()
  plt.scatter(x=[x[0] for x in epsilon_averages], y=[x[2] for x in epsilon_averages], c='blue', alpha=0.5, label='e')
  plt.scatter(x=[x[0] for x in p_averages], y=[x[2] for x in p_averages], c='red', alpha=0.5, label='p')
  plt.xlabel('TV(Pre,Post)')
  plt.ylabel('H(post)/H(Pre)')
  plt.legend()
  plt.title(args.dataset_name + ' TV')
  plt.savefig(args.name + 'plt-avg-ratio{}.png'.format(h_range))
  plt.clf()


