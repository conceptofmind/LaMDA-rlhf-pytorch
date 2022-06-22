## LaMDA-pytorch
Open-source implementation of Google's [LaMDA research paper](https://arxiv.org/abs/2201.08239) in PyTorch. The totally not sentient AI. This repository will cover the 2B paramater implementation of the model as that is likely what most can afford to train. You can review Google's latetst blog post from 2022 which details LaMDA [here](https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html). You can also view their previous blog post from 2021 on the model [here](https://blog.google/technology/ai/lamda/).

## Acknowledgement:
I have been greatly inspired by the brilliant code of [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains). Please check out his [open-source implementations](https://github.com/lucidrains) of multiple different transformer architectures and [support](https://github.com/sponsors/lucidrains) his work.

## Usage
```python
lamda_base = LaMDA(
    num_tokens = 20000,
    dim = 512,
    dim_head = 64,
    depth = 12,
    heads = 8
)

lamda = AutoregressiveWrapper(lamda_base, max_seq_len = 2048)

tokens = torch.randint(0, 20000, (1, 2048)) # mock token data

logits = lamda(tokens)

print("Cross-entropy Loss:", logits)
```

## About LaMDA:
- T5 Relative Positional Bias in Attention
- Gated GELU Activation in the Feed forward layer
- GPT-like Decoder Only architecture
- Autoregressive with Top-k sampling
- Sentencepiece Byte-pair encoded tokenizer

## TODO:
- [x] Finish building model architecture
- [ ] Add pre-training script
- [ ] Integrate Huggingface datasets
- [ ] Use [The Pile](https://github.com/EleutherAI/the-pile) from [Eleuther AI](https://github.com/EleutherAI). 
- [ ] Implement Sentencepiece tokenizer
- [ ] Add better documentation
- [ ] Add logging with [Weights And Biases](https://wandb.ai/site)
- [ ] Add [DeepSpeed](https://www.deepspeed.ai/) for scaling the model
- [ ] Add finetuning script
- [ ] Add pip installer with PyPI
- [ ] Implement a JAX / Flax version as well
- [ ] Add inference only if someone wants to open-source LaMDA model weights

## Citations
```bibtex
@article{DBLP:journals/corr/abs-2201-08239,
  author    = {Romal Thoppilan and
               Daniel De Freitas and
               Jamie Hall and
               Noam Shazeer and
               Apoorv Kulshreshtha and
               Heng{-}Tze Cheng and
               Alicia Jin and
               Taylor Bos and
               Leslie Baker and
               Yu Du and
               YaGuang Li and
               Hongrae Lee and
               Huaixiu Steven Zheng and
               Amin Ghafouri and
               Marcelo Menegali and
               Yanping Huang and
               Maxim Krikun and
               Dmitry Lepikhin and
               James Qin and
               Dehao Chen and
               Yuanzhong Xu and
               Zhifeng Chen and
               Adam Roberts and
               Maarten Bosma and
               Yanqi Zhou and
               Chung{-}Ching Chang and
               Igor Krivokon and
               Will Rusch and
               Marc Pickett and
               Kathleen S. Meier{-}Hellstern and
               Meredith Ringel Morris and
               Tulsee Doshi and
               Renelito Delos Santos and
               Toju Duke and
               Johnny Soraker and
               Ben Zevenbergen and
               Vinodkumar Prabhakaran and
               Mark Diaz and
               Ben Hutchinson and
               Kristen Olson and
               Alejandra Molina and
               Erin Hoffman{-}John and
               Josh Lee and
               Lora Aroyo and
               Ravi Rajakumar and
               Alena Butryna and
               Matthew Lamm and
               Viktoriya Kuzmina and
               Joe Fenton and
               Aaron Cohen and
               Rachel Bernstein and
               Ray Kurzweil and
               Blaise Aguera{-}Arcas and
               Claire Cui and
               Marian Croak and
               Ed H. Chi and
               Quoc Le},
  title     = {LaMDA: Language Models for Dialog Applications},
  journal   = {CoRR},
  volume    = {abs/2201.08239},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.08239},
  eprinttype = {arXiv},
  eprint    = {2201.08239},
  timestamp = {Fri, 22 Apr 2022 16:06:31 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2201-08239.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
```bibtex
@misc{https://doi.org/10.48550/arxiv.1706.03762,
  doi = {10.48550/ARXIV.1706.03762},
  
  url = {https://arxiv.org/abs/1706.03762},
  
  author = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Attention Is All You Need},
  
  publisher = {arXiv},
  
  year = {2017},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
