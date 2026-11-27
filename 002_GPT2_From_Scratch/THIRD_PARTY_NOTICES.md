# Third-party notices

## LLMs-from-scratch

Parts of the attention, transformer block and pretrained weight mapping structure were adapted from Sebastian
Raschka's *Build a Large Language Model (From Scratch)* and its accompanying repository:

- repository: `https://github.com/rasbt/LLMs-from-scratch`;
- revision checked during adaptation: `ec0bac5e1d854306a8ea6c308da49aae22479bd1`;
- copyright: 2023-2026 Sebastian Raschka;
- licence: Apache License 2.0, reproduced in `licenses/LLMs-from-scratch-Apache-2.0.txt`.

Submitted files are modified. Changes include package imports, mandatory causal masking, padding masks, tied
embeddings, a Transformers compatible causal-LM interface, cached generation, safe serialization, stricter source
coverage and new numerical tests.

## GPT-2 checkpoint

The package converts pretrained GPT-2 weights from the official Hugging Face mirror without redistributing them:

- model: `openai-community/gpt2`, also available through the `gpt2` alias;
- revision: `607a30d783dfa663caf39e06633721c8d4cfcd7e`;
- original software copyright: 2019 OpenAI;
- licence: [Modified MIT License](https://github.com/openai/gpt-2/blob/master/LICENSE); the checkpoint
  weights are loaded at runtime and are not redistributed with this package.

Transformers is used as the numerical reference and runtime model contract. Its source is not copied into this
package.
