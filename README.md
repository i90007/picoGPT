
# picoGPT

![nanoGPT](assets/nanogpt.jpg)

This is the small Llm from this [repo](https://github.com/KellerJordan/modded-nanogpt)

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to load pretrained model's state and run `sample.py`

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

## troubleshooting

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
