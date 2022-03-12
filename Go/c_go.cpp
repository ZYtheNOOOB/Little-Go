#include "c_go.h"

using namespace std;

void copy_arr(vector<vector<int>> &from, vector<vector<int>> &to) {
    for (int i=0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            to[i][j] = from[i][j];
        }
    }
}

void GO::set_board(int side, vector<vector<int>> &prev_board, vector<vector<int>> &cur_board) {
    for (int i=0; i < 5; i++) {
        for (int j=0; j < 5; j++) {
            if (prev_board[i][j] == side and cur_board[i][j] != side) {
                Point p = {.x = i, .y = j};
                died_pieces.push_back(p);
            }
        }
    }

    copy_arr(prev_board, previous_board);
    copy_arr(cur_board, board);
}

GO GO::copy_board() {
    GO copy_go;
    copy_go.n_move = this->n_move;
    for (auto &p : died_pieces) {
        copy_go.died_pieces.push_back(p);
    }
    copy_arr(previous_board, copy_go.previous_board);
    copy_arr(board, copy_go.board);
    return copy_go;
}

vector<Point> GO::detect_neighbor(int i, int j) {
    vector<Point> neighbors;
    if (i > 0) { Point p = {.x = i - 1, .y = j}; neighbors.push_back(p); }
    if (i < 4) { Point p = {.x = i + 1, .y = j}; neighbors.push_back(p); }
    if (j > 0) { Point p = {.x = i, .y = j - 1}; neighbors.push_back(p); }
    if (j < 4) { Point p = {.x = i, .y = j + 1}; neighbors.push_back(p); }
    return neighbors;
}

vector<Point> GO::detect_neighbor_ally(int i, int j) {
    vector<Point> allies;
    vector<Point> neighbors = detect_neighbor(i, j);
    for (auto &p : neighbors) {
        if (board[i][j] == board[p.x][p.y]) {
            allies.push_back(p);
        }
    }
    return allies;
}

vector<Point> GO::ally_dfs(int i, int j) {
    vector<Point> allies;
    Point root = {.x = i, .y = j};
    vector<Point> stack;
    stack.push_back(root);
    while (!stack.empty()) {
        Point piece = stack[stack.size() - 1];
        stack.pop_back();
        allies.push_back(piece);
        vector<Point> neighbor_allies = detect_neighbor_ally(piece.x, piece.y);
        for (auto &p : neighbor_allies) {
            bool traversed = false;
            for (auto &m : stack) {
                if (p.x == m.x and p.y == m.y) { traversed = true; break; }
            }
            for (auto &m : allies) {
                if (p.x == m.x and p.y == m.y) { traversed = true; break; }
            }
            if (!traversed) {
                stack.push_back(p);
            }
        }
    }
    return allies;
}

bool GO::find_liberty(int i, int j) {
    vector<Point> allies = ally_dfs(i, j);
    for (auto &p : allies) {
        vector<Point> neighbors = detect_neighbor(p.x, p.y);
        for (auto &m : neighbors) {
            if (board[m.x][m.y] == 0) {
                return true;
            }
        }
    }
    return false;
}

void GO::remove_died_pieces(int side) {
    vector<Point> died;
    for (int i=0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (board[i][j] == side) {
                if (!find_liberty(i, j)) {
                    Point p = {.x = i, .y = j};
                    died.push_back(p);
                }
            }
        }
    }
    // remove
    vector<vector<int>> board_dup {5, vector<int> {0, 0, 0, 0, 0}};
    copy_arr(board, board_dup);
    for (auto &p : died) {
        board_dup[p.x][p.y] = 0;
    }
    update_board(board_dup);

    vector<Point>().swap(died_pieces);
    if (!died.empty()) {
        for (auto &p : died) {
            died_pieces.push_back(p);
        }
    }
}

void GO::update_board(vector<vector<int>> &new_board) {
    for (int i=0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            board[i][j] = new_board[i][j];
        }
    }
}

bool GO::place_chess(int i, int j, int side) {
    copy_arr(board, previous_board);
    board[i][j] = side;
    return true;
}

bool GO::valid_place_check(int i, int j, int side) {
    if (board[i][j] != 0) { return false; }

    GO test_go = this->copy_board();
    test_go.board[i][j] = side;
    if (test_go.find_liberty(i, j)) { return true; }

    test_go.remove_died_pieces(3 - side);
    if (!test_go.find_liberty(i, j)) { return false; }
    else {
        if (!died_pieces.empty() and compare_board(previous_board, test_go.board)) { return false; }
    }

    return true;
}

bool GO::compare_board(vector<vector<int>> &b1, vector<vector<int>> &b2) {
    for (int i=0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (b1[i][j] != b2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

bool GO::game_end(int side, int action) {
    if (n_move >= max_move) { return true; }
    if (compare_board(previous_board, board) and action == 0) { return true; }
    return false;
}

int GO::score(int side) {
    int cnt = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (board[i][j] == side) {
                cnt++;
            }
        }
    }
    return cnt;
}

int GO::judge_winner() {
    auto cnt1 = (float) score(1);
    auto cnt2 = (float) score(2);
    if (cnt1 > cnt2 + komi) {
        return 1;
    } else if (cnt1 < cnt2 + komi) {
        return 2;
    } else {
        return 0;
    }
}

void GO::place_pass() {
    copy_arr(board, previous_board);
}

vector<Point> GO::valid_moves(int side) {
    vector<Point> moves;
    if (n_move == 0) {
        moves.push_back(Point{1, 1});
        moves.push_back(Point{1, 2});
        moves.push_back(Point{2, 2});
    } else if (n_move == 1) {
        for (int i=1; i<4; i++) {
            for (int j=1; j<4; j++) {
                if (board[i][j] == 0) { moves.push_back(Point{i, j}); }
            }
        }
    } else {
        for (int i=0; i<5; i++) {
            for (int j=0; j<5; j++) {
                if (valid_place_check(i, j, side)) { moves.push_back(Point{i, j}); }
            }
        }
        if (moves.empty()) { moves.push_back(Point{-1, -1}); }
    }
    return moves;
}

//int main() {
//    vector<vector<int>> prev = {{0, 0, 2, 0, 0}, {2, 1, 1, 2, 0}, {1, 0, 1, 1, 0}, {2, 1, 1, 0, 0}, {0, 2, 2, 0, 0}};
//    vector<vector<int>> cur = {{0, 0, 2, 0, 0}, {2, 1, 1, 2, 0}, {0, 2, 1, 1, 0}, {2, 1, 1, 0, 0}, {0, 2, 2, 0, 0}};
//    GO go;
//    go.set_board(1, prev, cur);
//    for (const auto& p : go.board) {
//        for (auto i : p) {
//            cout << i;
//        }
//        cout << endl;
//    }
//
//    for (int i = 0; i < 5; i++) {
//        for (int j = 0; j < 5; j++) {
//            if (go.valid_place_check(i, j, 1)) {
//                cout << "valid:" << " " << i << " " << j << endl;
//            }
//        }
//    }
//}
