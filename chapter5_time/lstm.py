"""
Long Short Term Memory (LSTM) <link https://ieeexplore.ieee.org/abstract/document/6795963 link> is a kind of recurrent neural network that can capture long-short term information.
This document mainly includes:
- Pytorch implementation for LSTM.
- An example to test LSTM.
For beginners, you can refer to <link https://zhuanlan.zhihu.com/p/32085405 link> to learn the basics about how LSTM works.
"""
from typing import Optional, Union, Tuple, List, Dict
import math
import torch
import torch.nn as nn
from ding.torch_utils.network.rnn import is_sequence
from ding.torch_utils import build_normalization


class LSTM(nn.Module):
    """
    **Overview:**
        Implementation of LSTM cell with layer norm.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            norm_type: Optional[str] = 'LN',
            dropout: float = 0.
    ) -> None:
        # Initialize arguments.
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialize normalization functions.
        norm_func = build_normalization(norm_type)
        self.norm = nn.ModuleList([norm_func(hidden_size * 4) for _ in range(2 * num_layers)])
        # Initialize LSTM parameters.
        self.wx = nn.ParameterList()
        self.wh = nn.ParameterList()
        dims = [input_size] + [hidden_size] * num_layers
        for l in range(num_layers):
            self.wx.append(nn.Parameter(torch.zeros(dims[l], dims[l + 1] * 4)))
            self.wh.append(nn.Parameter(torch.zeros(hidden_size, hidden_size * 4)))
        self.bias = nn.Parameter(torch.zeros(num_layers, hidden_size * 4))
        # Initialize the Dropout Layer.
        self.use_dropout = dropout > 0.
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout)
        self._init()

    # Dealing with different types of input and return preprocessed prev_state.
    def _before_forward(self, inputs: torch.Tensor, prev_state: Union[None, List[Dict]]) -> torch.Tensor:
        seq_len, batch_size = inputs.shape[:2]
        # If prev_state is None, it indicates that this is the beginning of a sequence. In this case, prev_state will be initialized as zero.
        if prev_state is None:
            zeros = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inputs.dtype,
                device=inputs.device
            )
            prev_state = (zeros, zeros)
        # If prev_state is not None, then preprocess it into one batch.
        else:
            assert len(prev_state) == batch_size
            state = [[v for v in prev.values()] for prev in prev_state]
            state = list(zip(*state))
            prev_state = [torch.cat(t, dim=1) for t in state]

        return prev_state

    def _init(self):
        # Initialize parameters. Each parameter is initialized using a uniform distribution of: $$U(-\sqrt {\frac 1 {HiddenSize}}, -\sqrt {\frac 1 {HiddenSize}})$$
        gain = math.sqrt(1. / self.hidden_size)
        for l in range(self.num_layers):
            torch.nn.init.uniform_(self.wx[l], -gain, gain)
            torch.nn.init.uniform_(self.wh[l], -gain, gain)
            if self.bias is not None:
                torch.nn.init.uniform_(self.bias[l], -gain, gain)

    def forward(self,
                inputs: torch.Tensor,
                prev_state: torch.Tensor,
                ) -> Tuple[torch.Tensor, Union[torch.Tensor, list]]:
        # The shape of input is: [sequence length, batch size, input size]
        seq_len, batch_size = inputs.shape[:2]
        prev_state = self._before_forward(inputs, prev_state)

        H, C = prev_state
        x = inputs
        next_state = []
        for l in range(self.num_layers):
            h, c = H[l], C[l]
            new_x = []
            for s in range(seq_len):
                # Calculate $$z, z^i, z^f, z^o$$ simultaneously.
                gate = self.norm[l * 2](torch.matmul(x[s], self.wx[l])
                                        ) + self.norm[l * 2 + 1](torch.matmul(h, self.wh[l]))
                if self.bias is not None:
                    gate += self.bias[l]
                gate = list(torch.chunk(gate, 4, dim=1))
                i, f, o, z = gate
                # $$z^i = \sigma (Wx^ix^t + Wh^ih^{t-1})$$
                i = torch.sigmoid(i)
                # $$z^f = \sigma (Wx^fx^t + Wh^fh^{t-1})$$
                f = torch.sigmoid(f)
                # $$z^o = \sigma (Wx^ox^t + Wh^oh^{t-1})$$
                o = torch.sigmoid(o)
                # $$z = tanh(Wxx^t + Whh^{t-1})$$
                z = torch.tanh(z)
                # $$c^t = z^f \odot c^{t-1}+z^i \odot z$$
                c = f * c + i * z
                # $$h^t = z^o \odot tanh(c^t)$$
                h = o * torch.tanh(c)
                new_x.append(h)
            next_state.append((h, c))
            x = torch.stack(new_x, dim=0)
            # Dropout layer.
            if self.use_dropout and l != self.num_layers - 1:
                x = self.dropout(x)
        next_state = [torch.stack(t, dim=0) for t in zip(*next_state)]
        # Return list type, split the next_state .
        h, c = next_state
        batch_size = h.shape[1]
        next_state = [torch.chunk(h, batch_size, dim=1), torch.chunk(c, batch_size, dim=1)]
        next_state = list(zip(*next_state))
        next_state = [{k: v for k, v in zip(['h', 'c'], item)} for item in next_state]
        return x, next_state


def test_lstm():
    # Randomly generate test data.
    seq_len = 2
    num_layers = 3
    input_size = 4
    hidden_size = 5
    batch_size = 6
    norm_type = 'LN'
    dropout = 0.1
    input = torch.rand(seq_len, batch_size, input_size).requires_grad_(True)
    lstm = LSTM(input_size, hidden_size, num_layers, norm_type, dropout)

    # Test the LSTM recurrently, using the hidden states of last input as new prev_state.
    prev_state = None
    for s in range(seq_len):
        input_step = input[s:s + 1]
        output, prev_state = lstm(input_step, prev_state)

    # Check whether the output is correct.
    assert output.shape == (1, batch_size, hidden_size)
    assert len(prev_state) == batch_size
    assert prev_state[0]['h'].shape == (num_layers, 1, hidden_size)
    torch.mean(output).backward()
    assert isinstance(input.grad, torch.Tensor)


if __name__ == '__main__':
    test_lstm()
