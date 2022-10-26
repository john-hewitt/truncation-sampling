import torch
import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette(n_colors=5)
in_c, out_c, p_c, e_c, h_c = colors

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
argp.add_argument('--sample_count', type=int, default = 4)
argp.add_argument('--generated', type=bool, default = False)
argp.add_argument('--dataset_name', default = 'webtext')
argp.add_argument('--max_length', type=int, default = 512)
argp.add_argument('--prefix_length', type=int, default=1)
argp.add_argument('--epsilon', type=float, default=0.0003)
argp.add_argument('--eta', type=float, default=0.0006)
argp.add_argument('--p', type=float, default=.95)
argp.add_argument('--model_string', default = 'gpt2-large')
argp.add_argument('--device', default = 'cuda')
argp.add_argument('--seed', default=-1, type=int)
argp.add_argument('--name', default='out')
argp.add_argument('--cache_dir', default=None)
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


entropy_counts = {x: 0 for x in [(0,.5), (0.5,1), (1,2), (2,4), (4,6), (6,100)]}

font = {
        'weight': 'normal',
        'size': 16,
        }

def possibly_plot(gold_indices, topp_warper, epsilon_warper, eta_warper, entropy, probs, logits, str_index):
  for entropy_range in entropy_counts.keys():
    if entropy > entropy_range[0] and entropy < entropy_range[1]:
      entropy_count = entropy_counts[entropy_range]
      entropy_counts[entropy_range] += 1
      break
  if entropy_count >= 20:
    return
  unwarped_probs = logits.softmax(dim=-1)
  eta_warped_probs, effective_eps = eta_warper(None, logits, return_epsilon=True)
  effective_eps = effective_eps.cpu().item()
  topp_warped_probs = topp_warper(None, logits)
  epsilon_warped_probs = epsilon_warper(None, logits)
  eta_cutoff_index = torch.sum(torch.logical_not(eta_warped_probs.isinf())).cpu().item()
  epsilon_cutoff_index = torch.sum(torch.logical_not(epsilon_warped_probs.isinf())).cpu().item()
  topp_cutoff_index = torch.sum(torch.logical_not(topp_warped_probs.isinf())).cpu().item()

  print(entropy, effective_eps, eta_cutoff_index, topp_cutoff_index)
  print('word, eta, topp')
  eta_tokens = []
  epsilon_tokens = []
  topp_tokens = []
  most_likely = []
  for index, word_id in enumerate(sorted(range(50256), key=lambda x:-unwarped_probs[0][x])): #range(50256):
    eta_accepted = not(bool(eta_warped_probs.reshape(-1)[word_id].isinf()))
    topp_accepted = not(bool(topp_warped_probs.reshape(-1)[word_id].isinf()))
    if index < 10:
      most_likely.append(word_id)
    if (index+1) > (eta_cutoff_index - 10) and (index+1) <= eta_cutoff_index:
      eta_tokens.append(word_id)
    if (index+1) > (topp_cutoff_index - 10) and (index+1) <= topp_cutoff_index:
      topp_tokens.append(word_id)
  print((tokenizer.decode(gold_indices),))
  print('most likely')
  for word_id in most_likely:
    print(tokenizer.decode(word_id).replace('\n', '\\n'))
  print('eta')
  for word_id in eta_tokens:
    print(tokenizer.decode(word_id).replace('\n', '\\n'))
  for word_id in eta_tokens:
    print("{:.2g}".format(unwarped_probs[0][word_id].item()))
  print('topp')
  for word_id in topp_tokens:
    print(tokenizer.decode(word_id).replace('\n', '\\n')) 
  for word_id in topp_tokens:
    print("{:.2g}".format(unwarped_probs[0][word_id].item()))

  plt.clf()
  plt.figure(figsize=(6.4, 3.2))
  y, _ = torch.sort(probs.reshape(-1).cpu(), descending=True)
  x = list(range(1, len(y)+1))
  cutoff = int(max(eta_cutoff_index, topp_cutoff_index, epsilon_cutoff_index)*1.2)
  x = x[:cutoff]
  y = y[:cutoff]
  ymin =0
  ymax=1
  plt.yscale('log')
  plt.bar(x=x, height=y, width=1)
  plt.xlabel('Rank of probability of word',fontdict=font)
  plt.ylabel('Probability of word',fontdict=font)
  plt.tight_layout()
  plt.savefig(str(str_index) + '-nolines.png', dpi=200)

  plt.axvline(x=topp_cutoff_index+.5, ymin=ymin, ymax=ymax, linestyle='--',linewidth=3, color=p_c, label='top-p cutoff')
  plt.axvline(x=eta_cutoff_index+.5, ymin=ymin, ymax=ymax, linestyle='--',linewidth=3, color=h_c, label='eta cutoff')
  plt.axvline(x=epsilon_cutoff_index+.5, ymin=ymin, ymax=ymax, linestyle='--',linewidth=3, color=e_c, label='epsilon cutoff')
  plt.legend(fontsize=16)
  print()
  print()

  plt.tight_layout()
  plt.savefig(str(str_index) + '.png', dpi=200)

topp_warper = warpers.TopPLogitsWarper(args.p)
eta_warper = warpers.EntropyWarper(args.eta)
epsilon_warper = warpers.EpsilonWarper(args.epsilon)

for str_index, string in enumerate([
  '<|endoftext|>My name X',
  '<|endoftext|>My name is X',
  '<|endoftext|>Donald X',
  '<|endoftext|>The capital of of the USA is Washington D.C. The capital of India is New Delhi. The capital of the UK is London. The capital of Ghana is X',
  '<|endoftext|>The X',
  '<|endoftext|>The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! The feeling! X']):
  gold_input_ids = tokenizer(string, return_tensors='pt').input_ids.to(args.device)[:,:args.max_length]
  model_outputs = model(input_ids=gold_input_ids)
  logits = model_outputs.logits
  index = gold_input_ids.size()[-1]-1
  last_word_logits = logits[:,index-1,:]
  unwarped_probs = torch.softmax(last_word_logits, dim=-1)
  orig_entropy = torch.distributions.Categorical(probs=unwarped_probs).entropy()
  possibly_plot(gold_input_ids[0,:index], topp_warper, epsilon_warper, eta_warper, orig_entropy, unwarped_probs, last_word_logits, str_index)
