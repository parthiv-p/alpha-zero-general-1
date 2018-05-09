'''
Board class for the game of TicTacToe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[2][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the board for the game of Othello by Eric P. Nichols.

'''
# from bkcharts.attributes import color
class Board():

    # list of all 8 directions on the board, as (x,y) offsets
    __directions = [(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]

    def __init__(self):
        "Set up initial board configuration."

        self.nx = 6
                 
        self.ny=7
        # Create the empty board array.
        self.pieces = [None]* self.nx
        for i in range(self.nx):
            self.pieces[i] = [0]*self.ny

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self, color):
        """Returns all the legal moves for the given color.
        (1 for white, -1 for black)
        @param color not used and came from previous version.        
        """
        moves = set()  # stores the legal moves.

        # Get all the empty squares (color==0)
        for y in range(self.ny):
            empty= -1
            for x in range(self.nx):
                if self[x][y]==0:
                    empty= x
                
            if empty==-1:
                pass
            else:   
                newmove = (empty,y)
                moves.add(newmove)
        return list(moves)

    def has_legal_moves(self):
        for y in range(self.ny):
            for x in range(self.nx):
                if self[x][y]==0:
                    return True
        return False
    
    def is_win(self, color):
        """Check whether the given player has collected a triplet in any direction; 
        @param color (1=white,-1=black)
        """
        win = 4     # need four in a row to win the game
        # check y-strips
        for y in range(self.ny):
            count = 0
            for x in range(self.nx):
                if self[x][y]==color:
                    count += 1
                    if count==win:
                        return True
                else:
                    count=0
        # check x-strips
        for x in range(self.nx):
            count = 0
            for y in range(self.ny):
                if self[x][y]==color:
                    count += 1
                    if count==win:
                        return True
                else:
                    count=0
                    
        # check two diagonal strips
        count = 0
        for i in range(self.ny):
            for d in range(self.nx):
                if d<self.nx and (d+i)<self.ny:
                    if self[d][d+i]==color:
                        count += 1
                        if count==win:
                            return True
                    else:              
                        count = 0

        count = 0
        for i in range(self.nx):
            for d in range(self.nx):
                if (d+i)<self.nx and (d)<self.ny:
                    if self[d+i][d]==color:
                        count += 1
                        if count==win:
                            return True
                    else:              
                        count = 0

        # check two diagonal strips
        count = 0
        for i in range(self.ny):
            for d in range(self.nx):
                if d<self.nx and (self.ny-1-(d+i))>=0:
                    if self[d][self.ny-1-(d+i)]==color:
                        count += 1
                        if count==win:
                            return True
                    else:              
                        count = 0

        count = 0
        for i in range(self.nx):
            for d in range(self.nx):
                if (d+i)<self.nx and (self.ny-1-(d))>=0:
                    if self[d+i][self.ny-1-(d)]==color:
                        count += 1
                        if count==win:
                            return True
                    else:              
                        count = 0
        return False

    def execute_move(self, move, color):
        """Perform the given move on the board; 
        color gives the color pf the piece to play (1=white,-1=black)
        """

        (x,y) = move

        # Add the piece to the empty square.
        assert self[x][y] == 0
        self[x][y] = color


