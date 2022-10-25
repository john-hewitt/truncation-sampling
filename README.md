# truncation-sampling

This repository describes experiments for the paper
_Truncation Sampling as Language Model Desmoothing_, comparing and evaluating
existing truncation sampling methods like top-p (nucleus), typical decoding
, and new methods epsilon sampling and our proposed eta sampling.

## Getting started

Start by installing some packages:

        pip install -r requirements.txt

## Finding MAUVE-maximizing hyperparameters

In many experiments, we set the truncation severity hyperparameter of each method
by maximizing the MAUVE score on open-ended WebText (in-distribution for the
GPT-2 models tested.) For those generations and results, and to replicate the
experiments, go to [our fork of the MAUVE paper repository](https://github.com/john-hewitt/ts-mauve-experiments).

## Human evaluation 
In these experiments, we took samples generated by GPT-2 large from prefixes
in the WebText data according to different truncation sampling techniques
and asked humans to state their preferences between them. In particular,
to test long document plausibility, we give humans the shared prefix, as well
as the last 70 words generated by two truncation sampling methods for that
prefix (or the real human-written suffix) and ask which suffix more plausibly
came from the same document as the prefix.
See our paper for more details and a screenshot of our mturk study. In the
`human_eval` subdirectory, we provide the results and the analysis scripts
for this portion of the study.

Note that we follow the MAUVE authors in skipping low-quality prefixes
in the WebText distribution when choosing the set of prefixes to generate
from.

See the `README.md` in the `human_eval` directory for more details.

## Automatic evaluations

### Repetition Analysis

To run the repetition analysis (Section 5.4), run

        python src/simple_repetition.py

### Entropy Analysis

To run the entropy analysis (Section 5.3), run

        python src/entropy_tv_tradeoff.py

### Individual distribution Analysis

To run the individual distribution analysis (Section 5.4), run

        python src/make_cutoff_plots.py
