import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional Encoding for draft order"""
    def __init__(self, d_model:int, max_len:int = 24):
        super().__init__()

        position=torch.arange(max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000.0)/d_model))

        pe=torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2]= torch.sin(position*div_term)
        pe[:, 0, 1::2]=torch.cos(position*div_term)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        #x: [batch, seq_len, d_model]
        return x + self.pe[:x.size(1), 0, :].unsqueeze(0)

class DraftModel(nn.Module):
    """Complete draft prediction model"""

    def __init__(self,
                 num_heroes:int=124,
                 embedding_dim:int=128,
                 num_layers:int=4,
                 num_heads:int=8,
                 dropout:float=0.2):
        super().__init__()

        #Hero embedding (125 to include padding token at index 0)
        self.hero_embedding=nn.Embedding(125, embedding_dim, padding_idx=0)

        #Positional Encoding
        self.pos_encoding=PositionalEncoding(embedding_dim)

        #Transformer
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer=nn.TransformerEncoder(encoder_layer, num_layers)

        #Policy Head (pick/ban prediction)
        self.policy_head=nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_heroes)
        )

        #Value Head (win probability)
        self.value_head=nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self._init_weights()
    
    def _init_weights(self):
        """Initialise weights"""
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, hero_sequence, valid_actions):

        #Create padding mask
        padding_mask=(hero_sequence==0)
        #Embed heroes
        x=self.hero_embedding(hero_sequence) #[batch, seq_len, embed_dim]
        #Add positional encoding
        x=self.pos_encoding(x)
        #Transform
        x=self.transformer(x, src_key_padding_mask=padding_mask)

        #Extract draft representation
        lengths=(~padding_mask).sum(dim=1)-1
        batch_indices=torch.arange(x.size(0), device=x.device)
        draft_repr=x[batch_indices, lengths.clamp(min=0)]

        #Generate predictions
        action_logits=self.policy_head(draft_repr)
        win_prob=self.value_head(draft_repr)

        #Mask invalid actions
        action_logits=action_logits.masked_fill(~valid_actions, float('-inf'))

        return action_logits, win_prob

