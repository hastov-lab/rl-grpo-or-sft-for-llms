# run switches live here, single source of truth for the choice type settings
from enum import Enum, auto


class Env(Enum):
    LOCAL = auto()
    COLAB = auto()


class Mode(Enum):
    SMOKE = auto()   # tiny check, few steps, no real compute
    FULL = auto()    # the real run


class Scale(Enum):
    M124 = auto()
    M355 = auto()
    M774 = auto()
    GPT2_B15 = auto()
    B05 = auto()
    B15 = auto()
    B3 = auto()
    B7 = auto()


class Condition(Enum):
    SFT_GRPO = auto()   # sft then grpo, the default pipeline
    BASE_GRPO = auto()  # grpo straight on the pretrained base (rq3)
    SFT_ONLY = auto()   # stop after sft
    MCSFT_GRPO = auto() # grpo on top of an external mc-sft adapter (experiment 028)
    EXTERNAL_SFT_GRPO = auto()  # GRPO on top of an archived external SFT stage
    MATCHED_LORA_GRPO = auto()  # matched rank-16 arm for experiment 035
    MATCHED_FULL_GRPO = auto()  # matched full-parameter arm for experiment 035
