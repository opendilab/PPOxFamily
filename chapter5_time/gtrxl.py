"""
Gated Transformer XL (GTrXL) <link https://arxiv.org/abs/1910.06764 link> is a stabilized transformer architecture for reinforcement learning.
This document mainly includes:
- Pytorch implementation for GTrXL.
- An example to test GTrXL.
"""
from typing import Optional, Dict
import warnings
import numpy as np
import torch
import torch.nn as nn
import treetensor
from ding.torch_utils import GRUGatingUnit, build_normalization

from ding.torch_utils.network.nn_module import fc_block
from ding.torch_utils.network.gtrxl import PositionalEmbedding, Memory, AttentionXL


class GatedTransformerXLLayer(torch.nn.Module):
    """
    **Overview:**
        Attention layer of GTrXL
    """
    def __init__(
            self,
            input_dim: int,
            head_dim: int,
            hidden_dim: int,
            head_num: int,
            mlp_num: int,
            dropout: nn.Module,
            activation: nn.Module,
            gru_gating: bool = True,
            gru_bias: float = 2.
    ) -> None:
        super(GatedTransformerXLLayer, self).__init__()
        self.dropout = dropout
        # Decide whether to use GRU-gating.
        self.gating = gru_gating
        if self.gating is True:
            self.gate1 = GRUGatingUnit(input_dim, gru_bias)
            self.gate2 = GRUGatingUnit(input_dim, gru_bias)
        # Build attention block using the AttentionXL class,
        # a feed-forward network with optional dropout, and two layer normalization layers.
        self.attention = AttentionXL(
            input_dim,
            head_dim,
            head_num,
            dropout,
        )
        # Build Feed-Forward-Network.
        layers = []
        dims = [input_dim] + [hidden_dim] * (mlp_num - 1) + [input_dim]
        for i in range(mlp_num):
            layers.append(fc_block(dims[i], dims[i + 1], activation=activation))
            if i != mlp_num - 1:
                layers.append(self.dropout)
        layers.append(self.dropout)
        self.mlp = nn.Sequential(*layers)
        # Build layer norm.
        self.layernorm1 = build_normalization('LN')(input_dim)
        self.layernorm2 = build_normalization('LN')(input_dim)
        self.activation = activation

    def forward(
            self,
            inputs: torch.Tensor,
            pos_embedding: torch.Tensor,
            u: torch.nn.Parameter,
            v: torch.nn.Parameter,
            memory: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Concat memory with input across sequence dimension. The shape is: [full_sequence, batch_size, input_dim]
        full_input = torch.cat([memory, inputs], dim=0)
        # Forward calculation for GTrXL layer.
        # In GTrXL, the layer normalization is put before the attention layer.
        x1 = self.layernorm1(full_input)
        # Attention module.
        a1 = self.dropout(self.attention(inputs, pos_embedding, x1, u, v, mask=mask))
        a1 = self.activation(a1)
        # In GTrXL, gating layer replace the resnet layer in TrXL.
        o1 = self.gate1(inputs, a1) if self.gating else inputs + a1


        x2 = self.layernorm2(o1)
        # Feed Forward Network.
        m2 = self.dropout(self.mlp(x2))
        o2 = self.gate2(o1, m2) if self.gating else o1 + m2
        return o2


class GTrXL(nn.Module):
    """
    **Overview:**
        PyTorch implementation for GTrXL.
    """
    def __init__(
        self,
        input_dim: int,
        head_dim: int = 128,
        embedding_dim: int = 256,
        head_num: int = 2,
        mlp_num: int = 2,
        layer_num: int = 3,
        memory_len: int = 64,
        dropout_ratio: float = 0.,
        activation: nn.Module = nn.ReLU(),
        gru_gating: bool = True,
        gru_bias: float = 2.,
        use_embedding_layer: bool = True,
    ) -> None:
        super(GTrXL, self).__init__()
        assert embedding_dim % 2 == 0, 'embedding_dim={} should be even'.format(input_dim)
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.embedding_dim = embedding_dim
        if isinstance(input_dim, list):
            input_dim = np.prod(input_dim)
        # Initialize embedding layer.
        self.use_embedding_layer = use_embedding_layer
        if self.use_embedding_layer:
            self.embedding = fc_block(input_dim, embedding_dim, activation=activation)
        # Initialize activate function.
        self.activation = activation
        # Initialize position embedding.
        self.pos_embedding = PositionalEmbedding(embedding_dim)
        # Memory to save hidden states of past segments. It will be initialized in the forward method to get its size dynamically.
        self.memory = None
        self.memory_len = memory_len
        # Initialize GTrXL layers.
        layers = []
        # Put all the embedding_dims into a list.
        # For the i-th layer, the input embedding is dims[i], while the output embedding is dims[i+1]
        dims = [embedding_dim] + [embedding_dim] * layer_num
        self.dropout = nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        for i in range(layer_num):
            layers.append(
                GatedTransformerXLLayer(
                    dims[i], head_dim, dims[i+1], head_num, mlp_num, self.dropout, self.activation, gru_gating,
                    gru_bias
                )
            )
        self.layers = nn.Sequential(*layers)
        # u and v are the parameters to compute global content bias and global positional bias.
        self.u, self.v = (
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
            torch.nn.Parameter(torch.zeros(self.head_num, self.head_dim)),
        )
        # Create an attention mask for each different seq_len. In this way we don't need to create a new one each time we call the forward method.
        self.att_mask = {}
        # Create a pos embedding for each different seq_len. In this way we don't need to create a new one each time we call the forward method.
        self.pos_embedding_dict = {}

    def reset_memory(self, batch_size: Optional[int] = None, state: Optional[torch.Tensor] = None):
        # Reset the memory of GTrXL.
        self.memory = Memory(memory_len=self.memory_len, layer_num=self.layer_num, embedding_dim=self.embedding_dim)
        # If batch_size is not None, specify the batch_size when initializing the memory.
        if batch_size is not None:
            self.memory = Memory(self.memory_len, batch_size, self.embedding_dim, self.layer_num)
        # If state is not None, add state into the memory.
        elif state is not None:
            self.memory.init(state)

    def get_memory(self):
        # Get the memory of GTrXL.
        if self.memory is None:
            return None
        else:
            return self.memory.get()

    def forward(self, x: torch.Tensor, batch_first: bool = False, return_mem: bool = True) -> Dict[str, torch.Tensor]:
        # If the first dimension of input x is batch_size,
        # then reshape x from  [batch_size ,sequence_length ,input_dim] to [sequence_length, batch_size, input_dim]
        if batch_first:
            x = torch.transpose(x, 1, 0)
        cur_seq, bs = x.shape[:2]
        # Get back memory.
        memory = None if self.memory is None else self.memory.get()
        # Abnormal case: no memory or memory shape mismatch.
        if memory is None:
            self.reset_memory(bs)
        elif memory.shape[-2] != bs or memory.shape[-1] != self.embedding_dim:
            warnings.warn(
                "Memory {} and Input {} dimensions don't match,"
                " this will cause the memory to be initialized to fit your input!".format(
                    list(memory.shape[-2:]), [x.shape[-2]] + [self.embedding_dim]
                )
            )
            self.reset_memory(bs)
        self.memory.to(x.device)
        memory = self.memory.get()
        # Pass through embedding layer.
        if self.use_embedding_layer:
            x = self.dropout(self.embedding(x))
        # Get full sequence length: memory length + current length
        prev_seq = self.memory_len
        full_seq = cur_seq + prev_seq
        # If the attention mask for current sequence length is already created, reuse the mask stored in self.att_mask.
        if cur_seq in self.att_mask.keys():
            attn_mask = self.att_mask[cur_seq]
        # Otherwise, create a new attention mask and store it into self.att_mask.
        else:
            # For example, if cur_seq = 3, full_seq = 7, then the mask is:
            # $$ \begin{matrix} 0 & 0 & 0 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 \end{matrix}$$
            # This forces that the hidden state of current token is only associated with previous tokens.
            attn_mask = (
                torch.triu(
                    torch.ones((cur_seq, full_seq)),
                    diagonal=1 + prev_seq,
                ).bool().unsqueeze(-1).to(x.device)
            )
            self.att_mask[cur_seq] = attn_mask
        # If the position encoding for current sequence length is already created, reuse it stored in self.pos_embedding_dict.
        if cur_seq in self.pos_embedding_dict.keys():
            pos_embedding = self.pos_embedding_dict[cur_seq]
        # Otherwise, create a new position encoding and store it into self.pos_embedding_dict.
        else:
            pos_ips = torch.arange(full_seq - 1, -1, -1.0, dtype=torch.float)  # full_seq
            pos_embedding = self.pos_embedding(pos_ips.to(x.device))
            self.pos_embedding_dict[cur_seq] = pos_embedding
        pos_embedding = self.dropout(pos_embedding)  # full_seq x 1 x embedding_dim

        hidden_state = [x]
        out = x
        # Calculate results for each GTrXL layer.
        for i in range(self.layer_num):
            layer = self.layers[i]
            out = layer(
                out,
                pos_embedding,
                self.u,
                self.v,
                mask=attn_mask,
                memory=memory[i],
            )
            hidden_state.append(out.clone())
        out = self.dropout(out)
        # Update the GTrXL memory.
        self.memory.update(hidden_state)
        # If the first dimension of output is required to be batch_size, then reshape x from  [sequence_length, batch_size, input_dim] to [batch_size ,sequence_length ,input_dim].
        if batch_first:
            out = torch.transpose(out, 1, 0)
        # Return memory is needed.
        if return_mem:
            output = treetensor.Object({"logit": out, "memory": memory})
        else:
            output = treetensor.Object({"logit": out})
        return output


def test_gtrxl() -> None:
    # Generate data for testing.
    input_dim = 128
    seq_len = 64
    bs = 32
    embedding_dim = 256
    layer_num = 5
    mem_len = 40
    memory = [None, torch.rand(layer_num + 1, mem_len, bs, embedding_dim)]

    # Test GTrXL under different situations.
    for i in range(2):
        m = memory[i]
        model = GTrXL(
            input_dim=input_dim,
            head_dim=2,
            embedding_dim=embedding_dim,
            memory_len=mem_len,
            head_num=2,
            mlp_num=2,
            layer_num=layer_num,
        )
        # Input shape: [sequence_length, batch_size, input_dim]
        input = torch.rand(seq_len, bs, input_dim, requires_grad=True)
        # Reset the model memory.
        if m is None:
            model.reset_memory(batch_size=bs)
        else:
            model.reset_memory(state=m)
        output = model(input)
        # Check the shape of output.
        assert output['logit'].shape == (seq_len, bs, embedding_dim)
        assert output['memory'].shape == (layer_num + 1, mem_len, bs, embedding_dim)
        torch.sum(output['logit']).backward()
        # Check the gradient.
        assert isinstance(input.grad, torch.Tensor)
        # Check memory.
        memory_out = output['memory']
        if m is not None:
            assert torch.all(torch.eq(memory_out, m))


if __name__ == '__main__':
    test_gtrxl()
