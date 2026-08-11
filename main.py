import torch
import torch.nn as nn
import torch.nn.functional as F

pip install tiktoken
# pip install tokenizers for self-built tokeniser

torch.manual_seed(0)
# training data 
text = """
Harry met the Minister's yellow eyes and knew he had no option but to obey. He held out his hand and Scrimgeour leaned forwards again and placed the Snitch, slowly and deliberately, into Harry's palm.
Nothing happened. As Harry's fingers closed around the snitch, its tired wings fluttered and were still. Scrimgeour, Ron and Hermione continued to gaze avidly at the now partially concealed ball, as if still hoping it might transform in some way.
'That was dramatic,' said Harry coolly. Both Ron and Hermione laughed. 
'That's all then, is it?' asked Hermione, making to prise herself off the sofa.
'Not quite,' said Scrimgeour, who looked bad-tempered now. 'Dumbledore left you a second bequest, Potter.'
'What is it?' asked Harry, excitement rekindling. 
Scrimgeour did not bother to read from the will this time.
'The sword of Godric Gryffindor,' he said. 
Hermione and Ron both stiffened. Harry looked around for a sign of the ruby-encrusted hilt, but Scrimgeour did not pull the sword from the leather pouch which, in any case, looked much too small to contain it.
""" * 20
# prebuilt subword tokeniser import
import tiktoken
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

def encode(s):
    return enc.encode(s)

def decode(ids):
    return enc.decode(ids)

data = torch.tensor(encode(text), dtype=torch.long)
# splitting into training/validating
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[:n]

# hyperparameters
d_model = 64
n_heads = 4
n_layers = 2
block_size = 64
batch_size = 32

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(0,len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+ block_size + 1]for i in ix])
    return x, y

# model

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 *d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x) :
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v, = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1,2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim = -1)

        out = att @ v
        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, d_model, n_heads) :
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp == nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU()
            nn.Linear(4 * d_model, d_model),
        )

