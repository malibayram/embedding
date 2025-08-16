import torch


class TokenizerMatcher:
  def __init__(self, source_tokenizer, target_tokenizer, source_model, target_model):
    self.source_tokenizer = source_tokenizer
    self.target_tokenizer = target_tokenizer
    self.source_model = source_model
    self.target_model = target_model

  def match_tokens(self) -> dict[int, list[int]]:
    vocab = self.target_tokenizer.get_vocab()
    tokens_map = {}

    for key, value in vocab.items():
      if value not in tokens_map:
        tokens_map[value] = []

      ids = self.source_tokenizer.encode(key)
      if 2 in ids:
        # remove bos token
        ids.remove(2)

      if len(tokens_map[value]) == 0 or len(tokens_map[value]) > len(ids):
        tokens_map[value] = ids

    return tokens_map

  def add_embeddings(self, ids) -> torch.Tensor:
    embeddings = []
    for id in ids:
      embeddings.append(self.source_model.model.embed_tokens(torch.tensor([id])))
    return torch.sum(torch.stack(embeddings), dim=0)

  def match_embeddings(self):
    matched_tokens = self.match_tokens()
    matched_embeddings = {}
    for key, value in matched_tokens.items():
      embedding = self.add_embeddings(value)
      matched_embeddings[key] = embedding
    return matched_embeddings
  
  def change_target_model_embeddings(self, matched_embeddings):
    with torch.no_grad():
      for key, value in matched_embeddings.items():
        self.target_model.embedder.weight[key] = value