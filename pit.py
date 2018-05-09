import Arena
from MCTS import MCTS
from connect4Team15.ConnectFourGame import ConnectFourGame, display
from connect4Team15.ConnectFourPlayers import *
from connect4Team15.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = ConnectFourGame()


hp = HumanConnectFourPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp-c4/','best.pth.tar')
args1 = dotdict({'numMCTSSims': 25, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))



arena = Arena.Arena(n1p, hp, g, display=display)
print(arena.playGames(2, verbose=True))