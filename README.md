## LaMDA-pytorch
Open-source implementation of Google's [LaMDA research paper](https://arxiv.org/abs/2201.08239) in PyTorch. The totally not sentient AI. This repository will cover the 2B paramater implementation of the model.

## In collaboration with:
- [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains)

## Notes about LaMDA:
- T5 Relative Positional Bias
- Gated GELU Activation
- GPT-like Decoder Only
- Autoregressive 
- Sentencepiece Byte-pair encoded tokenizer

## TODO:
- [ ] Finish building model architecture
- [ ] Add pre-training script
- [ ] Integrate Huggingface datasets
- [ ] Use [The Pile](https://github.com/EleutherAI/the-pile) from Eleuther AI 
- [ ] Implement Sentencepiece tokenizer
- [ ] Add better documentation
- [ ] Add logging with Weights And Biases
- [ ] Add [DeepSpeed](https://www.deepspeed.ai/) for scaling model
- [ ] Add finetuning script
- [ ] Add pip installer with PyPI
- [ ] Implement a JAX / Flax version as well
- [ ] Add inference only if someone wants to actually train a LaMDA model

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
