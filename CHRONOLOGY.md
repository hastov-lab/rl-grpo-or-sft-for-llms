# Experiment chronology

The dates below describe the 2026 experiment programme. Each link gives an example of the implementation or a saved experiment. The July smoke record was saved on 12 July. The first 3B run preceded the protocol freeze. Later packages included both planned comparisons and diagnostics prompted by earlier results.

| Dates | Work | Record |
| --- | --- | --- |
| 8 to 9 July | Implemented data preparation, overlap checks, training, evaluation and logging. | [000_LOCAL_SMOKE_B05_sft_grpo_seed0](001_Code/results/000_LOCAL_SMOKE_B05_sft_grpo_seed0) |
| 10 to 21 July | Refined the splits and ran the first 0.5B, 1.5B and 3B Qwen2.5 comparisons. | [split_audit_17Jul2026.json](001_Code/results/split_audit_17Jul2026.json) |
| 21 to 30 July | Extended the experiments to 7B, including a memory pilot, supervised controls and the remaining seed. | [013_COLAB_FULL_B7_sft_grpo_seed2](001_Code/results/013_COLAB_FULL_B7_sft_grpo_seed2) |
| 30 July to 4 August | Completed the remaining main seeds and the 3B Warm and Cold GRPO comparison. | [s4_comparison_summary.json](001_Code/results/s4_comparison_summary.json) |
| 7 to 21 August | Ran exposure, trajectory and reward checks and examined stopping rules, probe sensitivity, parameter updates, tiers and numerical representation. | [040_preflight.json](001_Code/results/040_preflight.json) |
| 21 August to 1 September | Verified GPT-2 pretrained weights and ran the four model sizes and the XL comparison. | [049_family_summary.json](001_Code/results/049_family_summary.json) |
| 9 to 11 September | Completed three checks of answer extraction and kept the original evaluations alongside them. | [052_COLAB_FULL_B05_sft_grpo_seed0](001_Code/results/052_COLAB_FULL_B05_sft_grpo_seed0) |
