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
      
      # Remove BOS and EOS tokens if present
      bos_id = self.source_tokenizer.bos_token_id
      eos_id = self.source_tokenizer.eos_token_id
      
      if bos_id is not None and bos_id in ids:
        ids.remove(bos_id)
      if eos_id is not None and eos_id in ids:
        ids.remove(eos_id)

      # Only update if we found a valid token sequence
      if len(ids) > 0:
        if len(tokens_map[value]) == 0 or len(tokens_map[value]) > len(ids):
          tokens_map[value] = ids

    return tokens_map

  def add_embeddings(self, ids, adding_style: str = "mean") -> torch.Tensor:
    if not ids:
      # Return zero embedding if no tokens found
      return torch.zeros(self.source_model.model.embed_tokens.embedding_dim, 
                        device=self.source_model.model.embed_tokens.weight.device)
    
    embeddings = []
    for id in ids:
      embedding = self.source_model.model.embed_tokens(torch.tensor([id], 
                                                                   device=self.source_model.model.embed_tokens.weight.device))
      embeddings.append(embedding)
    
    if adding_style == "mean":
      return torch.mean(torch.stack(embeddings), dim=0)
    elif adding_style == "sum":
      return torch.sum(torch.stack(embeddings), dim=0)
    else:
      raise ValueError(f"Invalid adding style: {adding_style}")

  def match_embeddings(self, adding_style: str = "sum"):
    matched_tokens = self.match_tokens()
    matched_embeddings = {}
    for key, value in matched_tokens.items():
      embedding = self.add_embeddings(value, adding_style)
      matched_embeddings[key] = embedding
    return matched_embeddings

  def change_target_model_embeddings(self, matched_embeddings):
    with torch.no_grad():
      target_device = self.target_model.embedder.weight.device
      changed_count = 0
      
      for key, value in matched_embeddings.items():
        # Ensure the embedding is on the same device as target model
        if value.device != target_device:
          value = value.to(target_device)
        
        # Check if the embedding is actually different before changing
        current_embedding = self.target_model.embedder.weight[key].clone()
        if not torch.allclose(current_embedding, value, atol=1e-6):
          self.target_model.embedder.weight[key] = value
          changed_count += 1
      
      print(f"Changed {changed_count} embeddings out of {len(matched_embeddings)} matched tokens")
      return changed_count
  
