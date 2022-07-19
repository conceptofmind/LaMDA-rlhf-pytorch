import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# CrossEntropyLoss

class LaMDA_Loss(nn.Module):
    def __init__(self):
        super(LaMDA_Loss, self).__init__()

    def forward(self, x_inp, x_labels, **kwargs):
        x_inp, x_labels = x_inp[:, :-1], x_labels[:, 1:]
        out = self.net(x_inp, **kwargs)
        loss = F.cross_entropy(rearrange(out, "b c n -> b n c"), x_labels)
        return loss

# autoregressive wrapper

def log(t, eps=1e-9):
    return torch.log(t + eps)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, max_seq_len = 1024, pad_value = 0):
        super().__init__()        
        self.pad_value = pad_value
        self.net = net
        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def generate(
        self, 
        start_tokens, 
        seq_len, 
        eos_token = None, 
        temperature = 1.0, 
        filter_logits_fn = top_k, 
        filter_thres = 0.9, 
        **kwargs
        ):
        
        was_training = self.net.training
        _, t = start_tokens.shape

        self.net.eval()
        out = start_tokens

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]

            logits = self.net(x, **kwargs)[:, -1, :]
            filtered_logits = filter_logits_fn(logits, thres = filter_thres)

            gumbel_noise = -log(-log(torch.zeros_like(filtered_logits).uniform_(0, 1)))
            sample = ((filtered_logits / temperature) + gumbel_noise).argmax(dim=-1)

            out = torch.cat((out, sample[:, None]), dim=-1)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]
        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        return self.net(x, **kwargs)