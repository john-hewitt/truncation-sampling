import torch
import transformers
import random

device = 'cuda'

class TypicalLogitsWarper(transformers.LogitsWarper):
    """
    Code largely taken from https://github.com/cimeister/typical-sampling
    """
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class TopPLogitsWarper(transformers.LogitsWarper):
  """
  [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off.
  Args:
      top_p (`float`):
          If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
          for generation.
      filter_value (`float`, *optional*, defaults to `-float("Inf")`):
          All filtered values will be set to this float value.
      min_tokens_to_keep (`int`, *optional*, defaults to 1):
          Minimum number of tokens that cannot be filtered.
  """

  def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
    top_p = float(top_p)
    if top_p < 0 or top_p > 1.0:
      raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

    self.top_p = top_p
    self.filter_value = filter_value
    self.min_tokens_to_keep = min_tokens_to_keep

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > self.top_p
    if self.min_tokens_to_keep > 1:
      # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
      sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

  def determine_divergence(self, scores, chosen_word):
    """
    Determines the extent to which a word violates the truncation threshold of this warper.

    Values above 0 indicate that the word would be disallowed (i.e., assigned zero probability) under this truncation method.

    Arguments:
        scores: Tensor of size (1, vocab_size)
        chosen_word: int
    Returns:
        how far past the top-p the word _before_ this in the sorted vocab is
        i.e., if the word _before_ this already accounted for p probability,
        the top-p truncation would have stopped there.
    """
    #import pdb; pdb.set_trace()
    sorted_logits, sorted_indices = torch.sort(scores, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    index = (sorted_indices == chosen_word).nonzero(as_tuple=True)[1]
    cum_pr = cumulative_probs[0,index]
    pr = cumulative_probs[0,index] - (cumulative_probs[0,index-1] if index > 1 else 0)
    return max(cum_pr-pr-self.top_p, 0)

class EtaWarper(transformers.LogitsWarper):
  """Our proposed eta sampling warper."""
  def __init__(self, epsilon):
    self.epsilon = epsilon
    self.filter_value = -float("Inf")

  def __call__(self, input_ids, scores) -> torch.FloatTensor:
    probabilities = scores.softmax(dim=-1)
    entropy = torch.distributions.Categorical(probs=(scores).softmax(dim=-1)).entropy()
    epsilon = min(self.epsilon, torch.sqrt(torch.tensor(self.epsilon))*torch.exp(-entropy))
    indices_to_remove = probabilities < epsilon
    max_word = torch.argmax(scores,dim=-1)
    indices_to_remove[...,max_word.squeeze()] = 0
    new_scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return new_scores

class EntropyWarper(transformers.LogitsWarper):
  """Same as EtaWarper; here for historical reasons."""
  def __init__(self, epsilon):
    self.epsilon = epsilon
    self.filter_value = -float("Inf")

  def __call__(self, input_ids, scores, return_epsilon=False) -> torch.FloatTensor:
    probabilities = scores.softmax(dim=-1)
    entropy = torch.distributions.Categorical(probs=(scores).softmax(dim=-1)).entropy()
    epsilon = min(self.epsilon, torch.sqrt(torch.tensor(self.epsilon))*torch.exp(-entropy))
    indices_to_remove = probabilities < epsilon
    max_word = torch.argmax(scores,dim=-1)
    indices_to_remove[...,max_word.squeeze()] = 0
    new_scores = scores.masked_fill(indices_to_remove, self.filter_value)
    if return_epsilon:
      return new_scores, torch.tensor(epsilon)
    else:
      return new_scores

class EpsilonWarper(transformers.LogitsWarper):
  """
  [`LogitsWarper`] that performs epsilon, i.e. restricting to tokens with absolute prob > prob_cut_off.
  Takes single argmax token if no tokens satisfy this constraint.
  Args:
      epsilon (`float`):
          If set to > 0, only the most tokens with probabilities `epsilon` or higher are kept for generation.
      filter_value (`float`, *optional*, defaults to `-float("Inf")`):
          All filtered values will be set to this float value.
  """
  def __init__(self, epsilon):
    self.epsilon = epsilon
    self.filter_value = -float("Inf")

  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    probabilities = scores.softmax(dim=-1)
    indices_to_remove = probabilities < self.epsilon
    max_word = torch.argmax(scores,dim=-1)
    indices_to_remove[...,max_word.squeeze()] = 0
    new_scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return new_scores

  def determine_divergence(self, scores, chosen_word):
    """
    Determines the extent to which a word violates the truncation threshold of this warper.

    Values above 0 indicate that the word would be disallowed (i.e., assigned zero probability) under this truncation method.

    Arguments:
        scores: Tensor of size (1, vocab_size)
        chosen_word: int
    Returns:
        how much less probability than (1-lmbda)*p_smoothing(chosen_word) 
        was assigned by this distribution to chosen_word; i.e., how much
        less likely it is than it would have needed to be if it were not
        to be truncated.
    """
    probabilities = scores.softmax(dim=-1)
    divergences = probabilities - self.epsilon
    return max(0, -divergences[0, chosen_word])


