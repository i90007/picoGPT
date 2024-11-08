
# picoGPT

![nanoGPT](assets/nanogpt.jpg)

I made this fork to make some improvements and add some new features to Karpathy's [project](https://github.com/karpathy/nanoGPT)

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to load pretrained weights and run `sample.py`

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

## todos

- ~~Add memorising like [here](https://colab.research.google.com/drive/1XZz1sjNt1MKRG6ul_hOGSJFQLS4lRtmJ).~~
- Teach the model to communicate in the form of a chat.
- Teach the model to communicate by voice as the main model of chatGPT. I'm going to add a seperate agent for a voice recognition (via OpenAI Swarm).
- Add separate agent/model for math (also via OpenAI Swarm).
- Add separate model for coding (OpenAI Swarm).
- Add web search feature.
- Teach Llm to pass SimpleQA benchmark.
- Make to work it locally like an app or like a web page.
- Add more functionality with separate small models (via Swarm) and make sure everything works fine on an usual phone.

## troubleshooting

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
