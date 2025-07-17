import copy
import torch

import torch.nn as nn
from typing import Optional


class DummySelfAttn:
    def __init__(self, batch_first: bool):
        self.batch_first = batch_first



def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayerBase(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation,batch_first):
        super(SPOTERTransformerDecoderLayerBase, self).__init__(d_model, nhead, dim_feedforward, dropout, activation,batch_first)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        #del self.self_attn
        # This atribute is added to maintain compatibility with the original TransformerDecoderLayer interface and latest torch versions.
        # It is not used in the forward method.
        object.__setattr__(self, 'self_attn', DummySelfAttn(batch_first))

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal = None,memory_is_causal=None) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes,hidden_dim=108, num_heads=9, num_layers_1=6, num_layers_2=6, 
                            dim_feedforward_encoder=64,
                            dim_feedforward_decoder=256,dropout=0.3,norm_first=False,batch_first=True):
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.pos         = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.class_query = nn.Parameter(torch.rand(1,1, hidden_dim))
        self.transformer  = nn.Transformer(d_model=hidden_dim, nhead =num_heads,
                                            num_encoder_layers= num_layers_1, 
                                            num_decoder_layers= num_layers_2,
                                            dim_feedforward = dim_feedforward_encoder,
                                            dropout=dropout,
                                            norm_first = norm_first,
                                            batch_first=batch_first)

        self.linear_class = nn.Linear(hidden_dim, num_classes)

        custom_decoder_layer = SPOTERTransformerDecoderLayerBase(self.transformer.d_model, self.transformer.nhead,
                                                             dim_feedforward_decoder, dropout=dropout, 
                                                             activation="relu",batch_first=batch_first)

        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def forward(self, x, padding_mask=None):

        batch_size, seq_len, feat_dim = x.size()
        class_query = self.class_query.expand(batch_size, -1, -1)# shape: (batch_size, 1, hidden_dim)
        h = self.transformer(src=self.pos+x,
                             tgt=class_query,
                             src_key_padding_mask=padding_mask)
        res = self.linear_class(h).squeeze(1)
        return res


if __name__ == "__main__":
    pass