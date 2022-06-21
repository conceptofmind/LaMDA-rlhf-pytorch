## LaMDA-pytorch
Open-source implementation of Google's LaMDA in PyTorch. The totally not sentient AI.

## In collaboration with:
- [Dr. Phil 'Lucid' Wang](https://github.com/lucidrains)

## Notes about architecture:
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
