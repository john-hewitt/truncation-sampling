import transformers
import json
import argparse
import torch
import os
from tqdm import tqdm
import random
import warpers
import requests



argp = argparse.ArgumentParser()
argp.add_argument('warper')
argp.add_argument('--seed', type=int, default=0)
argp.add_argument('--model_string', default='gpt2-large')
argp.add_argument('--cache_dir', default=None)
argp.add_argument('--device', default='cuda')
args = argp.parse_args()

torch.manual_seed(args.seed)

device = args.device

try:
  model = transformers.GPT2LMHeadModel.from_pretrained(args.model_string, cache_dir=args.cache_dir).to(device)
  tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_string, cache_dir=args.cache_dir)
except requests.exceptions.HTTPError:
  model = transformers.GPT2LMHeadModel.from_pretrained(args.model_string, cache_dir=args.cache_dir, local_files_only=True).to(device)
  tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.model_string, cache_dir=args.cache_dir, local_files_only=True)
model.config.pad_token_id = model.config.eos_token_id

model.eval()
for param in model.parameters():
  param.requires_grad = False


model.config.pad_token_id = model.config.eos_token_id

# instantiate logits processors
probs = []

if args.warper == 'p':
  if args.model_string == 'gpt2-xl':
    hyp = 0.95
  if args.model_string == 'gpt2-large':
    hyp = 0.95
  elif args.model_string == 'gpt2':
    hyp = 0.9
  elif args.model_string == 'gpt2-medium':
    hyp = 0.89
  warper_class = warpers.TopPLogitsWarper
elif args.warper == 'e':
  warper_class = warpers.EpsilonWarper
  if args.model_string == 'gpt2-xl':
    hyp = 0.0003
  if args.model_string == 'gpt2-large':
    hyp = 0.0003
  elif args.model_string == 'gpt2-medium':
    hyp = 0.0009
  elif args.model_string == 'gpt2':
    hyp = 0.0006
elif args.warper == 'h':
  warper_class = warpers.EtaWarper
  if args.model_string == 'gpt2-xl':
    hyp = 0.0003
  if args.model_string == 'gpt2-large':
    hyp = 0.0006
  elif args.model_string == 'gpt2-medium':
    hyp = 0.002
  elif args.model_string == 'gpt2':
    hyp = 0.002
elif args.warper == 'typ':
  warper_class = warpers.TypicalLogitsWarper
  if args.model_string == 'gpt2-xl':
    hyp = 0.92
  if args.model_string == 'gpt2-large':
    hyp = 0.92
  elif args.model_string == 'gpt2-medium':
    hyp = 0.9
  elif args.model_string == 'gpt2':
    hyp = 0.9


prompts = [json.loads(x) for x in open('data/repetition_prompts.jsonl')]

model.eval()
for param in model.parameters():
  param.requires_grad = False

pred_length = 512

data = []
logits_warper = transformers.LogitsProcessorList(
    [
      warper_class(hyp)
    ]
)
total = 0
repetition = 0
data = []
with open('repetition.{}.{}.{}'.format(args.warper, args.seed, args.model_string), 'w') as fout:
  for prompt in tqdm(prompts, desc='[prompts]'):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)[:,:35]
    input_ids = torch.cat([input_ids] + [input_ids[:,-4:]]*5, dim=1)
    print(tokenizer.decode(input_ids[0]))
    for i in tqdm(range(5)):
      outputs = model.sample(input_ids, logits_warper=logits_warper,
          stopping_criteria=transformers.StoppingCriteriaList([transformers.MaxLengthCriteria(max_length=pred_length+input_ids.size()[-1])]))
      model_outputs = model(input_ids=outputs, labels=outputs)
      if model_outputs.loss < 1:
        print()
        print(model_outputs.loss.item(), (tokenizer.batch_decode(outputs, skip_special_tokens=True)[0],))#.replace('\n', ' '))
        repetition += 1
      total += 1
      print('Hyperparam: {}, Total: {}, repetition: {}'.format(hyp, total, repetition))
      fout.write(json.dumps((model_outputs.loss.item(), tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0], tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])) + '\n')
