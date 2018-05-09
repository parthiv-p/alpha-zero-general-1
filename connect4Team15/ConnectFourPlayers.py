import numpy as np

"""
Random and Human-ineracting players for the game of TicTacToe.
Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.
Based on the OthelloPlayers by Surag Nair.
"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanConnectFourPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print(int(i/self.game.ny), int(i%self.game.ny))
        while True: 
            # Python 3.x
            a = input()
            # Python 2.x 
            # a = raw_input()

            x,y = [int(x) for x in a.split(' ')]
            a = self.game.ny * x + y if x!= -1 else self.game.nx*self.game.ny
            if valid[a]:
                break
            else:
                print('Invalid')

        return a
