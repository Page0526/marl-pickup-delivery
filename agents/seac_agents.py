import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import random
from agents.agent import Agents
import torch.optim as optim


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU()
class SEACAgent:
    def __init__(self):
        pass
    
    def init_agents(self):
        pass

