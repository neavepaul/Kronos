import chess
import numpy as np
import tensorflow as tf
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

from modules.athena.src.utils import (
    fen_to_tensor, 
    encode_move_sequence, 
    get_attack_defense_maps,
    encode_move,
    decode_move
)
from modules.athena.src.aegis_net import AegisNet
from modules.ares.logic.mcts import MCTS

class Athena:
    pass