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
