import transformers
import json
import random
import pickle

tok = transformers.GPT2Tokenizer.from_pretrained('gpt2')

#eta_strings = pickle.load(open('/u/scr/johnhew/jag-code/mauve-experiments/outputs/webtext_gpt2-large/generations/basic/sentences_test_p1.0_k0_t1.0_e0.0_h0.0006_seed1.p', 'rb'))[0]
#topp_strings = pickle.load(open('/u/scr/johnhew/jag-code/mauve-experiments/outputs/webtext_gpt2-large/generations/basic/sentences_test_p0.95_k0_t1.0_e0.0_h0.0_seed1.p', 'rb'))[0]
eta_strings = pickle.load(open('sentences_test_p1.0_k0_t1.0_e0.0_h0.0006_seed1.p', 'rb'))[0]
topp_strings = pickle.load(open('sentences_test_p0.95_k0_t1.0_e0.0_h0.0_seed1.p', 'rb'))[0]
gold_strings = [json.loads(x)['text'] for x in open('../../data/webtext.test.jsonl')]

total = 0
eta = 0

cuts_out = open('rejected_for_human_eval-test.jsonl', 'w')

with open('for_human_eval-test.jsonl', 'w') as fout:
  #for i, goldstring in zip(range(len(eta_strings)), data.get_dataset('webtext', 'test')):
  for i, goldstring in zip(range(len(eta_strings)), gold_strings):
   eta_tok = tok(eta_strings[i]).input_ids
   gold_tok = tok(goldstring).input_ids
   topp_tok = tok(topp_strings[i]).input_ids
   if (900 <= len(eta_tok) <= 1100) and (900 <= len(topp_tok) <= 1100) and (900 <= len(gold_tok) <= 1100):
     prefix = tok.decode(gold_tok[:35])
     eta_suffix = tok.decode(eta_tok[-70:])
     topp_suffix = tok.decode(topp_tok[-70:])
     gold_suffix = tok.decode(gold_tok[-70:])
     print(prefix)
     keep = bool(int(input()))
     if keep:
       fout.write(json.dumps({'eta': eta_strings[i], 'topp': topp_strings[i], 'human':goldstring,
                 'eta_suffix':eta_suffix, 'topp_suffix':topp_suffix, 'gold_suffix':gold_suffix, 'prefix':prefix})+'\n')
       total += 1
     else:
       cuts_out.write(json.dumps({'eta': eta_strings[i], 'topp': topp_strings[i], 'human':goldstring,
                 'eta_suffix':eta_suffix, 'topp_suffix':topp_suffix, 'gold_suffix':gold_suffix, 'prefix':prefix})+'\n')
  print(total)
