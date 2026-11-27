# GPT-2 from scratch

We implemented the GPT-2 architecture in PyTorch for the boundary experiments in this study. We initialised each model with OpenAI's pretrained weights and used it for supervised fine-tuning and GRPO at four sizes, with 124M, 355M, 774M and 1.5B parameters. The [training pipeline](../001_Code/train/model_utils.py) loads the model and tokenizer for each run.

## Implementation

- The [model](gpt2_scratch/model.py) combines token and position embeddings with transformer blocks and a language modelling head. The input embeddings and output head share weights. During training, the model predicts the next token and computes the loss against the shifted labels.
- [Causal attention](gpt2_scratch/attention.py) prevents a token from attending to later tokens and masks padding. During generation, the model can reuse the keys and values from earlier tokens. Models can be saved and reloaded with their tied weights intact.
- The [configuration](gpt2_scratch/configuration.py) defines the four architectures. The [weight loader](gpt2_scratch/weights.py) maps a pinned reference checkpoint into each one and checks tensor shapes during conversion.
- Training uses the Transformers GPT-2 fast tokenizer. The [tokenizer tests](tests/test_tokenizer_parity.py) check that it encodes and decodes fixed prompts the same way as the [tiktoken wrapper](gpt2_scratch/tokenizer.py).

## Install and test

Follow [Getting started](../README.md#getting-started) for the environment, editable installation and checkpoint cache setup. To run just the package tests, use this command from the repository root.

```bash
python -m pytest 002_GPT2_From_Scratch/tests
```

The [124M reference tests](tests/test_pretrained_parity.py) compare next-token scores and training loss with Transformers 5.5.4 at GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Other tests check [cached and greedy generation](tests/test_generation_cache.py) and [saving and reloading the model](tests/test_save_load.py). The [124M notebook](notebooks/001_gpt2_124m_parity.ipynb) walks through the comparison. The [scale comparison notebook](notebooks/002_gpt2_scale_parity.ipynb) runs the comparison for a selected model size.

## Source acknowledgements

Parts of the attention, transformer block and weight mapping code are adapted from Sebastian Raschka's *Build a Large Language Model (From Scratch)* and its accompanying repository. The [third-party notices](THIRD_PARTY_NOTICES.md) identify the upstream revision, copyright and licence for the adapted code and the original GPT-2 licence for the pretrained weights.
