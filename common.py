import os
import random

import numpy as np

ALGO_CHOICES = [
    "io",
    "sc",
    "cot",
    "tot",
    "minimax",
    "heuristic",
    "max_power",
    "dmg_calc",
    "random",
    "None",
]
MODEL_CHOICES = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "llama",
    "deepseek-chat",
    "None",
]
BATTLE_FORMAT_CHOICES = ["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"]

PNUMBER1 = 0
while True:
    if not os.path.exists(f"./llm_log/{PNUMBER1}"):
        os.makedirs(f"./llm_log/{PNUMBER1}")
        break
    PNUMBER1 += 1
print(PNUMBER1)
seed = 42
random.seed(seed)
np.random.seed(seed)
