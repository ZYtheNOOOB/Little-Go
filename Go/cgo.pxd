# distutils: sources = c_go.cpp

from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "c_go.cpp":
    cdef struct Point:
        int x
        int y

    cdef cppclass GO:
        int size, n_move, max_move
        float komi
        vector[Point] died_pieces
        vector[vector[int]] board
        vector[vector[int]] previous_board

        void set_board(int side, vector[vector[int]] prev_board, vector[vector[int]] cur_board)
        GO copy_board()
        void remove_died_pieces(int side)
        bool place_chess(int i, int j, int side)
        void place_pass()
        bool valid_place_check(int i, int j, int side)
        vector[Point] valid_moves(int side)
        void update_board(vector[vector[int]] new_board)
        bool game_end(int side, int action)
        bool compare_board(vector[vector[int]] b1, vector[vector[int]] b2)
        int score(int side)
        int judge_winner()
