# TimeSeriesSeq2Seq
The repository aims to give basic understandings on **time-series sequence-to-sequence(Seq2Seq) model** for beginners.  
The repo implements the following Seq2Seq models:  
- LSTM encoder-decoder
- LSTM encoder-decoder with attention by [Bahdanau et al.(2014)](https://arxiv.org/abs/1409.0473). See ```architecture.rnn```, ```architecture.attention``` and ```architecture.seq2seq.AttentionLSTMSeq2Seq```
- Vanilla Transformer by [Vaswani et al.(2017)](https://arxiv.org/abs/1706.03762). See ```architecture.Transformer``` and ```architecture.seq2seq.TransformerSeq2Seq```.  

> I only implemented three types of Seq2Seq model.  
> You may combine 1D CNN, LSTM and Transformer layers for further-customized version.

See [Tutorial.ipynb](https://github.com/hyeonbeenlee/NeuralSeq2Seq/blob/main/Tutorial.ipynb) for testing.  

# Configuration
Suppose $`(B,L_{in},C_{in})\xrightarrow{model}(B,L_{out},C_{out})`$ operation, where  

```math
\begin{aligned}
B&=\text{batch\_size}\\
L_{in}&=\text{input\_sequence\_length (variable)}\\
C_{in}&=\text{input\_embedding\_size}\\
L_{out}&=\text{output\_sequence\_length (variable)}\\
C_{out}&=\text{output\_embedding\_size}\\
\end{aligned}
```  

- ```hidden_size``` Hidden state size of LSTM encoder. Equivalent to ```d_model``` in ```TransformerSeq2Seq```  
- ```num_layers``` Number of LSTM and Transformer encoder, decoder layers.
- ```bidirectional``` Whether to use bidirectional LSTM encoder.  
- ```dropout``` Dropout rate. Applies to:  
  - Residual drop path in 1DCNN in ```architecture.cnn```  
  - Hidden state dropout in LSTM encoder/decoder(for every time step). Unlike ```torch.nn.LSTM```, dropout is applied from the first LSTM layer.  
  - The same dropout as Vanilla Transformer from Vaswani et al..
- ```layernorm``` Layer normalization in LSTM encoder and decoder.  

# Autoregressive mode and teacher-forced mode
All Seq2Seq models inherit ```architecture.seq2seq.Seq2Seq``` class:
```
class Seq2Seq(Skeleton):
    def __init__(self):
        super().__init__()
        self.initialize_skeleton(locals())

    def forward_auto(self, x: torch.Tensor, trg_len: int):
        # implements autoregressive decoding
        raise NotImplementedError

    def forward_labeled(self, x: torch.Tensor, y: torch.Tensor):
        # implements teacher-forced decoding
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        trg_len: int,
        y: Optional[torch.Tensor] = None,
        teacher_forcing: float = -1,
    ):
        batch_size, length_x, input_size = x.size()
        p = random.uniform(0, 1)
        if p > teacher_forcing and y is not None:
            return self.forward_labeled(x, y)
        else:
            return self.forward_auto(x, trg_len)
```
##### ```forward_auto```
implements autoregressive forward, which uses the model's previous time step output as current time step input in its decoder.  
- Generally used in inference stage, when **label output data is not available.**

##### ```forward_labeled```
implements teacher-forced forward, which uses previous time step label data as current time step input in the decoder.  
- Generally used in training stage, when label output data is available.  

##### ```forward``` 
runs ```forward_labeled``` with probability of ```teacher_forcing```.  
Else generates $`(\text{batch\_size},\ \text{trg\_len},\ C_{out})`$ shaped output tensor using ```forward_auto```.


# Set hyperparameters and dummy tensors

```
import torch
import torch.nn as nn
from architecture.seq2seq import LSTMSeq2Seq, AttentionLSTMSeq2Seq, TransformerSeq2Seq

batch_size = 32
length_x = 40  # input sequence length
length_y = 60  # output label sequence length
input_size = 27  # input feature size
output_size = 6  # output feature size
hidden_size = 128  # hidden state size of LSTM encoder. Equivalent to d_model in TransformerSeq2Seq
dropout = 0.1  # dropout rate
num_layers = 3  # number of LSTM and Transformer encoder, decoder layers
bidirectional = True  # Whether to use bidirectional LSTM encoder
layernorm = True  # Layer normalization in LSTM encoder and decoder

x = torch.randn(batch_size, length_x, input_size)
y = torch.randn(batch_size, length_y, output_size)
```

### LSTM Encoder-Decoder

```
model = LSTMSeq2Seq(
    input_size=input_size,
    output_size=output_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional,
    dropout=dropout,
)
out_1 = model.forward_auto(x, 100)
out_2 = model.forward_labeled(x, y)
print(out_1.shape, out_2.shape)
```
```
torch.Size([32, 100, 6]) torch.Size([32, 60, 6])
```

### LSTM Encoder-Decoder with Bahdanau style attention
```
model = AttentionLSTMSeq2Seq(
    input_size=input_size,
    output_size=output_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    bidirectional=bidirectional,
    dropout=dropout,
)
out_1 = model.forward_auto(x, 100)
out_2 = model.forward_labeled(x, y)
print(out_1.shape, out_2.shape)
```
```
torch.Size([32, 100, 6]) torch.Size([32, 60, 6])
```
### Vanilla Transformer Encoder-Decoder
```
model = TransformerSeq2Seq(
    input_size=input_size,
    output_size=output_size,
    num_layers=num_layers,
    d_model=hidden_size,
    n_heads=4,
    dropout=dropout,
    d_ff=hidden_size * 4,
)
out_1 = model.forward_auto(x, 100)
out_2 = model.forward_labeled(x, y)
print(out_1.shape, out_2.shape)
```
```
torch.Size([32, 100, 6]) torch.Size([32, 60, 6])
```
# Accessing model properties
- Parameters can be counted by ```model.count_params()```
- Properties are accessed using ```model.model_info``` attribute.  
- Another model instance can be created by ```ModelClass(**model.model_init_args)```.  

These features are attributed to ```architectures.skeleton.Skeleton``` class.
```
model.count_params()

model_info = model.model_info
model_init_args = model.model_init_args
print(model_info)

another_model_instance = TransformerSeq2Seq(**model_init_args)
```
```
Number of trainable parameters: 1,393,798
{'bidirectional': True, 'd_ff': 512, 'd_model': 128, 'dropout': 0.1, 'hidden_size': 256, 'input_size': 27, 'layernorm': False, 'n_heads': 4, 'num_hl': 0, 'num_layers': 3, 'output_size': 6}
```

# Parameter initialization
All network parameters are random-initialized from $\mathcal{N}\sim(0,0.01^2)$, except for all bias with ```torch.zeros``` and normalization layer weights with ```torch.ones```. See ```architectures.init```.
