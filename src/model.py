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

        #Hero embedding (num_heroes + 1 to include padding token at index 0)
        self.hero_embedding=nn.Embedding(num_heroes + 1, embedding_dim, padding_idx=0)

        #Type Embedding
        #2 types: 0=Ban, 1=Pick
        self.type_embedding=nn.Embedding(2, embedding_dim)

        #Team Embedding
        #2 teams: 0=Radiant, 1=Dire
        self.team_embedding=nn.Embedding(2, embedding_dim)

        #Positional Encoding
        #self.pos_encoding=PositionalEncoding(embedding_dim) we will not do this, instead we will use learnable position embeddings
        self.pos_embedding=nn.Embedding(24, embedding_dim)

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

        #Role Head (predict role for each hero in sequence)
        #Output: 6 classes (0=Unknown/Ban, 1-5=Positions)
        self.role_head=nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

        #Synergy Head
        #Catch roles, create lane pairs and mid pair. Calculate the gold difference
        synergy_input_dim=embedding_dim+(24*6)
        self.synergy_head=nn.Sequential(
            nn.Linear(synergy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3) #Safe Diff, Mid Diff, Off Diff
        ) 

        self._init_weights()
    
    def _init_weights(self):
        """Initialise weights"""
        for p in self.parameters():
            if p.dim()>1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, hero_sequence, type_sequence, team_sequence, valid_actions):

        #Create padding mask
        padding_mask=(hero_sequence==0)

        #Embed heroes
        hero_emb=self.hero_embedding(hero_sequence) #[batch, seq_len, embed_dim]
        type_emb=self.type_embedding(type_sequence)
        team_emb=self.team_embedding(team_sequence)

        #Combine Embeddings
        x=hero_emb+type_emb+team_emb

        #Add positional encoding
        positions=torch.arange(0, 24, device=x.device).unsqueeze(0)
        x=x+self.pos_embedding(positions)

        #Pass to transformer
        x=self.transformer(x, src_key_padding_mask=padding_mask)

        #Extract draft representation
        lengths=(~padding_mask).sum(dim=1)-1
        batch_indices=torch.arange(x.size(0), device=x.device)
        draft_repr=x[batch_indices, lengths.clamp(min=0)]

        #Generate predictions
        action_logits=self.policy_head(draft_repr)
        win_prob=self.value_head(draft_repr)
        role_logits=self.role_head(x) # [batch, seq_len, 6]

        #Use role logits as an intermediate, flatten them before use: [batch, 24, 6] -> [batch, 144]
        role_flat=role_logits.view(role_logits.size(0), -1)

        #Concatenate with draft repr
        synergy_input=torch.cat([draft_repr, role_flat], dim=1)

        synergy_preds=self.synergy_head(synergy_input)

        #Mask invalid actions
        action_logits=action_logits.masked_fill_(~valid_actions, float('-inf'))

        return action_logits, win_prob, role_logits, synergy_preds

