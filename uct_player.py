import os

import numpy as np
import random
import time

import go as g
from read import readInput
from write import writeOutput


class MCTSPlayer:
    def __init__(self, side, iter=1000, time=9.0, test_agent=False):
        self.mcts_iter = iter
        self.mcts_time = time
        self.exploration = 1
        self.side = side
        # memory: list of (state, reward)
        self.memory = []

        self.select_time = 0
        self.expand_time = 0
        self.rollout_time = 0
        self.backup_time = 0
        self.iter = 0
        self.test = test_agent

    def reward(self, result):
        if result == 0:
            return 0
        if result == self.side:
            return 1
        else:
            return -1

    def move(self, go, side):
        action = self.search(go=go, side=self.side)
        if action == 25:
            return 0
        else:
            return action // 5, action % 5

    def search(self, go, side):
        root = UCTNode(go, side)
        start_time = time.time()
        cur_time = time.time()
        i = 0
        while i < self.mcts_iter and cur_time - start_time < self.mcts_time:
            node = self.select(root)
            r = self.rollout(node, self.random_policy)
            self.backup(node, reward=r)
            i += 1
            self.iter += 1
            cur_time = time.time()
        return self.best_child(root)

    def select(self, node):
        while not node.terminal:
            if not node.expanded:
                return self.expand(node)
            else:
                start = time.time()
                best_key = self.best_child(node)
                node = node.children[best_key]
                end = time.time()
                self.select_time += end - start
        return node

    def best_child(self, node):
        values = np.array([c.value / c.visit + self.exploration * np.sqrt(2 * np.log2(node.visit) / c.visit)
                           for _, c in node.children.items()])
        if node.side == self.side:
            max_values = np.where(values == np.max(values))[0]
        else:
            max_values = np.where(values == np.min(values))[0]
        return list(node.children.keys())[np.random.choice(max_values)]

    def expand(self, node):
        start = time.time()
        # choose move
        move = random.choice(node.possible_moves)
        node.possible_moves.remove(move)
        if len(node.possible_moves) == 0:
            node.expanded = True

        # make move
        action = 1 if move[0] != -1 else 0
        child_go = node.go.copy_board()
        if action == 1:
            child_go.place_chess(move[0], move[1], node.side)
            child_go.remove_died_pieces(3 - node.side)
        child_go.n_move += 1

        # initialize new node
        child = UCTNode(child_go, 3 - node.side)
        # terminal
        if child_go.game_end(node.side, action):
            result = child_go.judge_winner()
            r = self.reward(result)
            child.terminal = True
            child.expanded = True
            child.reward = r
            child.value = r
            child.visit = 1

        if action == 1:
            child_go.place_pass()

        # add to children
        if action == 1:
            node.children[move[0] * 5 + move[1]] = child
        elif action == 0:
            node.children[25] = child
        child.parent = node

        end = time.time()
        self.expand_time += end - start
        return child

    def backup(self, node, reward):
        start = time.time()
        while node:
            node.visit += 1
            if not node.terminal:
                node.value += reward
            else:
                node.value = node.visit * node.reward
            node = node.parent
        end = time.time()
        self.backup_time += end - start

    def rollout(self, node, policy):
        if node.terminal:
            return node.reward
        else:
            r = self.reward(policy(node))
            return r

    def encode_state(self, go, side):
        # encode state
        prev_state_p1 = np.array(go.previous_board, dtype=np.float32).reshape((1, -1))
        prev_state_p1[prev_state_p1 == 2] = 0
        prev_state_p2 = np.array(go.previous_board, dtype=np.float32).reshape((1, -1))
        prev_state_p2[prev_state_p2 == 1] = 0

        curr_state_p1 = np.array(go.board, dtype=np.float32).reshape((1, -1))
        curr_state_p1[curr_state_p1 == 2] = 0
        curr_state_p2 = np.array(go.board, dtype=np.float32).reshape((1, -1))
        curr_state_p2[curr_state_p2 == 1] = 0

        plyr_turn = np.array([[1, 0]]) if side == 1 else np.array([[0, 1]])
        # print('rollout:')
        # print(prev_state.shape, curr_state.shape, plyr_turn.shape)
        state = np.concatenate((prev_state_p1, prev_state_p2, curr_state_p1, curr_state_p2, plyr_turn), axis=1)
        # print(state)
        # print(state.shape)
        return state

    # def valuenet_policy(self, node):
    #     input = self.encode_state(node.go, node.side)
    #     r = self.value_net(input)
    #     return np.squeeze(r)

    def random_policy(self, node):
        side = node.side
        go = node.go.copy_board()
        # print('start rolling')
        while True:
            possible_moves = [(m['x'], m['y']) for m in go.valid_moves(side)]
            move = random.choice(possible_moves)

            action = 1 if move[0] != -1 else 0
            if action == 1:
                go.place_chess(move[0], move[1], side)
                go.remove_died_pieces(3 - side)
            go.n_move += 1

            if go.game_end(side, action):
                break
            if action == 0:
                go.place_pass()
            side = 3 - side
        return go.judge_winner()


class UCTNode:
    def __init__(self, go, side):
        self.go = go
        self.board = go.board
        self.side = side
        # valid moves + pass
        # if h:
        #     self.possible_moves = get_valid_moves(self.go, self.side)
        # else:
        #     self.possible_moves = get_valid_moves_naive(self.go, self.side)
        self.possible_moves = [(m['x'], m['y']) for m in go.valid_moves(side)]

        self.visit = 0
        self.value = 0
        self.reward = 0
        self.children = {}
        self.parent = None
        self.expanded = False
        self.terminal = False


def read_moves():
    with open('n_move.txt', 'r') as f:
        n_move = int(f.readline())
    return n_move


def write_moves(n_move):
    with open('n_move.txt', 'w') as f:
        f.write(str(n_move))


# if __name__ == "__main__":
#     N = 5
#     piece_type, previous_board, board = readInput(N)
#     if os.path.exists('n_move.txt') and np.sum(previous_board) == 0:
#         os.remove('n_move.txt')
#
#     if os.path.exists('n_move.txt'):
#         n_move = read_moves()
#     else:
#         n_move = 1 if np.sum(board) > 0 else 0
#     go = GO(N)
#     go.set_board(piece_type, previous_board, board)
#     go.n_move = n_move
#     player = GoPlayer(side=piece_type, iter=5000)
#     action = player.move(go, piece_type)
#     writeOutput(action)
#     write_moves(n_move + 2)
