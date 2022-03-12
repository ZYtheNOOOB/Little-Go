# distutils: language = c++
from copy import deepcopy
from cgo cimport GO

cdef class PyGO:
    cdef GO go

    def set_board(self, side, prev, cur):
        return self.go.set_board(side, prev, cur)

    def valid_moves(self, side):
        return self.go.valid_moves(side)

    def place_chess(self, i, j, side):
        return self.go.place_chess(i, j, side)

    def place_pass(self):
        self.go.place_pass()

    def remove_died_pieces(self, side):
        self.go.remove_died_pieces(side)

    def game_end(self, side, action):
        if action != 0:
            action = 1
        return self.go.game_end(side, action)

    def judge_winner(self):
        return self.go.judge_winner()

    def copy_board(self):
        new_go = PyGO()
        new_go.go = self.go.copy_board()
        return new_go

    @property
    def board(self):
        return self.go.board

    @property
    def n_move(self):
        return self.go.n_move
    @n_move.setter
    def n_move(self, n):
        self.go.n_move = n
