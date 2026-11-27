# the tag-format instructions, kept in one place so the wording is easy to review
MMLU_INSTRUCTION = "Answer the multiple choice question. Think through it step by step inside <think></think> tags, then give only the letter of the correct option (A, B, C or D) inside <answer></answer> tags."
GSM8K_INSTRUCTION = "Answer the maths question. Think through it step by step inside <think></think> tags, then give only the final number inside <answer></answer> tags."

# fixed neutral think for the mc-sft control, matched to the vestigial thinks grpo learns
MC_SFT_THINK = "I will provide the selected option below."
