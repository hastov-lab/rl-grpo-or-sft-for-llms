# GRPO or supervised fine-tuning for language models

This repository accompanies the comparison of GRPO and supervised fine-tuning across Qwen2.5 and GPT-2 model sizes and MMLU subject difficulty tiers. The first release covers the saved checkpoints. The research code and experiment results will follow separately.

## Download checkpoints

Choose a model size and download every numbered part in its row. The 1,675 original files occupy 39.00 GB in total. Each archive restores the original paths under `checkpoints/`. The selection includes final adapters, full models, GPT-2 starting models and selected intermediate states. Some adapters refer to a pretrained model or a supervised starting model that must be loaded or reconstructed separately.

| Model | Checkpoint files | Size in GB | Download parts in order | Check restored files |
| --- | ---: | ---: | --- | --- |
| Qwen2.5 0.5B | 360 | 1.91 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-0.5b.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-0.5b.tar.part-002) | [SHA-256](manifests/qwen2.5-0.5b.sha256) |
| Qwen2.5 1.5B | 246 | 8.41 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-1.5b.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-1.5b.tar.part-002) · [3](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-1.5b.tar.part-003) · [4](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-1.5b.tar.part-004) · [5](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-1.5b.tar.part-005) | [SHA-256](manifests/qwen2.5-1.5b.sha256) |
| Qwen2.5 3B | 217 | 3.15 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-3b.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-3b.tar.part-002) | [SHA-256](manifests/qwen2.5-3b.sha256) |
| Qwen2.5 7B | 425 | 8.99 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-7b.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-7b.tar.part-002) · [3](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-7b.tar.part-003) · [4](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-7b.tar.part-004) · [5](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/qwen2.5-7b.tar.part-005) | [SHA-256](manifests/qwen2.5-7b.sha256) |
| GPT-2 124M | 127 | 3.59 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-124m.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-124m.tar.part-002) | [SHA-256](manifests/gpt2-124m.sha256) |
| GPT-2 355M | 64 | 1.61 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-355m.tar.part-001) | [SHA-256](manifests/gpt2-355m.sha256) |
| GPT-2 774M | 64 | 3.44 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-774m.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-774m.tar.part-002) | [SHA-256](manifests/gpt2-774m.sha256) |
| GPT-2 1.5B | 172 | 7.89 | [1](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-1.5b.tar.part-001) · [2](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-1.5b.tar.part-002) · [3](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-1.5b.tar.part-003) · [4](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-1.5b.tar.part-004) · [5](https://github.com/hastov-lab/rl-grpo-or-sft-for-llms/releases/download/checkpoints-v1/gpt2-1.5b.tar.part-005) | [SHA-256](manifests/gpt2-1.5b.sha256) |

### Restore one model size

For example, download the Qwen2.5 7B parts to an empty directory. From that directory, run the following commands on macOS or Linux. Replace `/path/to/repository` with the path to this repository. Allow space for both the downloaded parts and the extracted files.

```sh
grep '  qwen2.5-7b.tar.part-' /path/to/repository/manifests/release-parts.sha256 | shasum -a 256 -c -
cat qwen2.5-7b.tar.part-* | tar -xf -
shasum -a 256 -c /path/to/repository/manifests/qwen2.5-7b.sha256
```

Continue to extraction only after every part reports `OK`. The final command checks every restored study file against the original backup. To select another model, replace `qwen2.5-7b` with the model identifier from its download filenames. On Windows, use WSL to run the same commands, or use the download index to join the parts in numerical order in binary mode before extracting the TAR archive.

The [download index](manifests/downloads.json) records the exact byte count and SHA-256 for every numbered part. Individual checkpoint paths and checksums are listed in the model manifests above.

## Model licences

**Built with Qwen.** Qwen2.5 3B materials are for non-commercial research or evaluation under the [Qwen Research License Agreement](licenses/Qwen-Research.txt). The other three Qwen2.5 sizes use [Apache License 2.0](licenses/Qwen-Apache-2.0.txt). GPT-2 downloads carry the [original GPT-2 licence](licenses/GPT-2.txt). Each archive includes the applicable licence and [modification notes](licenses/MODIFICATIONS.md).
