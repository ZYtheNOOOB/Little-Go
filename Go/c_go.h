//
// Created by ZY on 2022/3/10.
//

#ifndef GO_C_GO_H
#define GO_C_GO_H

#include <vector>
#include <string>
#include <iostream>

using namespace std;

struct Point {
    int x;
    int y;
};

void copy_arr(vector<vector<int>> &from, vector<vector<int>> &to);

class GO {
public:
    int size = 5;
    int n_move = 0;
    int max_move = 24;
    float komi = 2.5;
    vector<Point> died_pieces;
    vector<vector<int>> board {5, vector<int> {0, 0, 0, 0, 0}};
    vector<vector<int>> previous_board {5, vector<int> {0, 0, 0, 0, 0}};

    void set_board(int side, vector<vector<int>> &prev_board, vector<vector<int>> &cur_board);
    GO copy_board();
    void remove_died_pieces(int side);
    bool place_chess(int i, int j, int side);
    void place_pass();
    bool valid_place_check(int i, int j, int side);
    vector<Point> valid_moves(int side);
    void update_board(vector<vector<int>> &new_board);
    bool game_end(int side, int action);
    bool compare_board(vector<vector<int>> &b1, vector<vector<int>> &b2);
    int score(int side);
    int judge_winner();

private:
    vector<Point> detect_neighbor(int i, int j);
    vector<Point> detect_neighbor_ally(int i, int j);
    vector<Point> ally_dfs(int i, int j);
    bool find_liberty(int i, int j);
};

#endif //GO_C_GO_H