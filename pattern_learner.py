import collections
import os
import random
import time

import numpy as np
import json

import go as g
from extractor import extract, encode
from read import readInput
from write import writeOutput

weight_file = 'weights.json'
weight_sharing_file = 'weight_sharing.json'
neutral_file = 'neutral.json'


class TDLearner:
    # weights
    weights = {}
    # neutral
    neutral = {'0000000000000000000000000': 1, '0': 1, '0000': 1, '000000000': 1}
    # weight sharing: encoded key -> (stored key, reversion)
    weight_sharing = {}
    # reward
    reward = {1: 1, 2: -1, 0: 0}

    def __init__(self, alpha=.2, max_iter=5000, max_time=9.0, epsilon=.1):
        self.alpha = alpha
        self.epsilon = epsilon

        # search params
        self.max_iter = max_iter
        self.max_time = max_time
        self.exploration = 5

        # history: (state value, feature key -> feature num)
        self.TD_history = collections.deque()

        self.select_time = 0
        self.expand_time = 0
        self.feature_time = 0
        self.move_time = 0

    def move(self, go, side, mode='test'):
        # if mode == 'test':
        return self.move2(go, side, mode)
        # else:
        #     return self.move1(go, side, mode)

    # def move1(self, go, side, mode='test'):
    #     """ make a move
    #     """
    #     # current state value
    #     state_value, features = self.compute_state_value(board=go.board)
    #     # epsilon greedy
    #     if mode == 'train' and np.random.rand() < self.epsilon:
    #         actions = [(i, j) for j in range(5) for i in range(5) if go.valid_place_check(i, j, side)] + [(-1, -1)]
    #         act = random.choice(actions)
    #         # add to history
    #         if act[0] == -1:
    #             self.TD_history.append((state_value, features))
    #         else:
    #             next_state_value, next_features = self.get_next_state_value(go, act[0], act[1], side)
    #             self.TD_history.append((next_state_value, next_features))
    #     else:
    #         actions = [(i, j) for j in range(5) for i in range(5) if go.valid_place_check(i, j, side)]
    #         action_values = [(state_value, features, (-1, -1))]
    #         # choose action
    #         for i, j in actions:
    #             next_state_value, next_features = self.get_next_state_value(go, i, j, side)
    #             action_values.append((next_state_value, next_features, (i, j)))
    #         # black: max, white: min
    #         if side == 1:
    #             action_values.sort(key=lambda x: x[0], reverse=True)
    #         else:
    #             action_values.sort(key=lambda x: x[0], reverse=False)
    #         act = random.choice([a for a in action_values if a[0] == action_values[0][0]])
    #         # add to history
    #         self.TD_history.append((act[0], act[1]))
    #         # return action only
    #         act = act[2]
    #
    #     if len(self.TD_history) > 2:
    #         self.TD_history.popleft()
    #
    #     if act[0] == -1:
    #         return "PASS"
    #     else:
    #         return act

    def move2(self, go, side, mode='test'):
        """ make a move
        """
        # current state value
        state_value, features = self.compute_state_value(board=go.board)
        # epsilon-greedy
        if mode == 'train' and np.random.rand() < self.epsilon:
            actions = [(m['x'], m['y']) for m in go.valid_moves(side)]
            act = random.choice(actions)
            # add to history
            if act[0] == -1:
                self.TD_history.append((state_value, features, state_value))
            else:
                next_state_value, next_features = self.get_next_state_value(go, act[0], act[1], side)
                self.TD_history.append((next_state_value, next_features, next_state_value))
        else:
            # search and gain short-term memory
            action, searched_iter, searched_time, Q = self.search(go, side)
            # print('searched', searched_iter, 'iterations in time:', searched_time, 's')
            # print(state_value, Q)
            # print('select:', self.select_time)
            # print('expand:', self.expand_time)
            # print('feature:', self.feature_time)
            # print('move:', self.move_time)
            if action != 25:
                act = (action // 5, action % 5)
            else:
                act = (-1, -1)

            # add to history
            if act[0] == -1:
                self.TD_history.append((state_value, features, Q))
            else:
                next_state_value, next_features = self.get_next_state_value(go, act[0], act[1], side)
                self.TD_history.append((next_state_value, next_features, Q))

        if len(self.TD_history) > 2:
            self.TD_history.popleft()

        if act[0] == -1:
            return 0
        else:
            return act

    def search(self, go, side):
        root = TreeNode(go, side, self)
        # print('valid moves:', root.moves)
        i = 0
        start_time = time.time()
        cur_time = time.time()
        while i < self.max_iter and cur_time - start_time < self.max_time:
            node = self.select(root)
            r = self.evaluate(node)
            self.backup(node, r=r)

            i += 1
            cur_time = time.time()

        # if i > 10000:
        #     queue = collections.deque([root])
        #     while len(queue) > 0:
        #         node = queue.popleft()
        #         if node.visit > 100:
        #             print('')
        #             node.go.visualize_board()
        #             print('step:', node.go.n_move)
        #             print('side:', node.side)
        #             print('Q:', node.Q())
        #             print('num visit:', node.visit)
        #             print('post value:', node.post_value)
        #             print('prior value:', node.prior_value)
        #             print('children:', get_valid_move(go, node.side))
        #         for k, c in node.children.items():
        #             queue.append(c)
        best_child = self.best_child(root, only_q=True)
        return best_child, i, cur_time - start_time, root.children[best_child].Q()

    def select(self, node):
        is_root = True

        while not node.terminal:
            if not node.expanded:
                return self.expand(node)
            else:
                start = time.time()
                best_key = self.best_child(node, is_root=is_root)
                node = node.children[best_key]
                end = time.time()
                self.select_time += end - start
            is_root = False
        return node

    def best_child(self, node, only_q=False, is_root=False):
        if only_q:
            values = np.array([c.Q() for _, c in node.children.items()])
        else:
            # compute PUCT value
            if is_root:
                n = len(node.children.items())
                d_noise = np.random.dirichlet(alpha=[0.3 for _ in range(n)], size=(n,))
                values = np.array(
                    [c.Q() + self.exploration * (0.75 * node.probs[i] + 0.25 * d_noise[i]) *
                     np.sqrt(node.visit) / (1 + c.visit) for i, (_, c) in enumerate(node.children.items())])
            else:
                values = np.array(
                    [c.Q() + self.exploration * node.probs[i] * np.sqrt(node.visit) / (1 + c.visit)
                     for i, (_, c) in enumerate(node.children.items())])
        # black: max, white: min
        if node.side == 1:
            max_values = np.where(values == np.max(values))[0]
        else:
            max_values = np.where(values == np.min(values))[0]
        return list(node.children.keys())[np.random.choice(max_values)]

    def expand(self, node):
        e_start = time.time()

        i, j = random.choice(node.moves)
        node.moves.remove((i, j))

        # make move
        action = 1 if i != -1 else 0
        child_go = node.go.copy_board()
        if action == 1:
            child_go.place_chess(i, j, node.side)
            child_go.remove_died_pieces(3 - node.side)
        child_go.n_move += 1

        # initialize new node
        child = TreeNode(child_go, 3 - node.side, self)
        # terminal
        if child_go.game_end(node.side, action):
            result = child_go.judge_winner()
            r = self.reward[result]
            child.terminal = True
            child.expanded = True
            child.reward = r
            child.post_value = r
            child.visit = 1

        if action == 0:
            child_go.place_pass()

        # add to children
        if action == 1:
            node.children[i * 5 + j] = child
        elif action == 0:
            node.children[25] = child
        child.parent = node

        if len(node.moves) == 0:
            # probs
            priors = np.array([c.prior_value for c in node.children.values()])
            prior_e = np.exp(priors - np.max(priors))
            prior_probs = prior_e / prior_e.sum()
            node.probs = prior_probs
            node.expanded = True

        end = time.time()
        self.expand_time += end - e_start
        return child

    def backup(self, node, r):
        while node:
            node.visit += 1
            if not node.terminal:
                node.post_value += r
            else:
                node.post_value = node.visit * r
            node = node.parent

    def evaluate(self, node):
        if node.terminal:
            return node.reward
        else:
            return node.prior_value

    def learn(self, r):
        if len(self.TD_history) < 2:
            return
        else:
            # compute TD-error
            value_t1, feature_t1, _ = self.TD_history[0]
            value_t2 = r if r else self.TD_history[1][2]
            td_error = (value_t2 - value_t1) / np.sum([n ** 2 for n in feature_t1.values()])
            # update
            for k, n in feature_t1.items():
                w_key, w_rev = self.weight_sharing[k]
                if not self.neutral.get(w_key):
                    delta = self.alpha * n * td_error * w_rev
                    weight = self.weights[w_key] + delta
                    self.weights[w_key] = weight

    def get_next_state_value(self, go, i, j, side):
        next_go = go.copy_board()
        next_go.place_chess(i, j, side)
        next_go.remove_died_pieces(3 - side)
        next_state_value, next_features = self.compute_state_value(board=next_go.board)
        return next_state_value, next_features

    def compute_state_value(self, board):
        start = time.time()

        feature, feature_raw = extract(np.array(board))

        end = time.time()
        self.feature_time += end - start

        feature_value = {}
        for k in feature.keys():
            # check weight sharing
            pattern = feature_raw[k]
            p_size = pattern.shape[0]
            # check existing mapping
            if self.weight_sharing.get(k):
                w_key, w_rev = self.weight_sharing[k]
                feature_value[k] = feature[k] * self.weights[w_key] * w_rev
            # not in existing mapping
            else:
                # initialize weight
                self.weights[k] = 0.
                feature_value[k] = 0.
                self.weight_sharing[k] = (k, 1)
                # add weight sharing for features except li 1x1
                if p_size != 1:
                    self.add_weight_sharing(pattern=pattern, p_size=p_size, shared_key=k)
                    # flip for 2x2 and 3x3 location independent
                    if p_size != 5:
                        # flip
                        p_flip = np.flipud(pattern)
                        k_flip = encode(p_flip, size=p_size)
                        self.weight_sharing[k_flip] = (k, 1)
                        self.add_weight_sharing(pattern=p_flip, p_size=p_size, shared_key=k)
                else:
                    # reverse only for 1x1
                    p_rev = self.reverse_shape(pattern)
                    k_rev = encode(p_rev, size=p_size)
                    self.weight_sharing[k_rev] = (k, -1)
        # tanh activation
        state_value = np.tanh(np.sum(list(feature_value.values())))

        return state_value, feature

    def add_weight_sharing(self, pattern, p_size, shared_key):
        # rotate
        k_rot90 = encode(np.rot90(pattern, 1), size=p_size)
        self.weight_sharing[k_rot90] = (shared_key, 1)
        k_rot180 = encode(np.rot90(pattern, 2), size=p_size)
        self.weight_sharing[k_rot180] = (shared_key, 1)
        k_rot270 = encode(np.rot90(pattern, 3), size=p_size)
        self.weight_sharing[k_rot270] = (shared_key, 1)
        # reverse
        p_rev = self.reverse_shape(pattern)
        k_rev = encode(p_rev, size=p_size)
        self.weight_sharing[k_rev] = (shared_key, -1)
        # reverse + rotate
        k_rev_rot90 = encode(np.rot90(p_rev, 1), size=p_size)
        self.weight_sharing[k_rev_rot90] = (shared_key, -1)
        k_rev_rot180 = encode(np.rot90(p_rev, 2), size=p_size)
        self.weight_sharing[k_rev_rot180] = (shared_key, -1)
        k_rev_rot270 = encode(np.rot90(p_rev, 3), size=p_size)
        self.weight_sharing[k_rev_rot270] = (shared_key, -1)
        # check neutral
        if encode(np.flipud(pattern), size=p_size) == k_rev or \
                encode(np.fliplr(pattern), size=p_size) == k_rev or k_rev_rot180 == shared_key:
            self.neutral[shared_key] = 1

    def reverse_shape(self, p):
        h, w = p.shape
        p_rev = np.zeros_like(p)
        for i in range(h):
            for j in range(w):
                if p[i, j] != 0:
                    p_rev[i, j] = 3 - p[i, j]
        return p_rev

    def save(self, path):
        with open(path+weight_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
        with open(path+weight_sharing_file, 'w') as f:
            json.dump(self.weight_sharing, f, indent=2)
        with open(path+neutral_file, 'w') as f:
            json.dump(self.neutral, f, indent=2)

    def load(self, path):
        with open(path+weight_file, 'r') as f:
            self.weights = json.load(f)
        with open(path+weight_sharing_file, 'r') as f:
            self.weight_sharing = json.load(f)
        with open(path+neutral_file, 'r') as f:
            self.neutral = json.load(f)


class TreeNode:
    def __init__(self, go, side, plyr):
        self.go = go
        self.side = side
        # given by search
        self.post_value = 0
        # given by long term memory
        self.prior_value, self.features = plyr.compute_state_value(go.board)
        # possible actions
        start = time.time()
        self.moves = [(m['x'], m['y']) for m in go.valid_moves(side)]
        end = time.time()
        plyr.move_time += end - start

        # prior probs
        self.probs = []

        self.reward = None
        self.visit = 0
        self.children = {}

        self.parent = None
        self.expanded = False
        self.terminal = False

    def Q(self):
        if self.visit == 0:
            return 0
        else:
            return self.post_value / self.visit


# def get_valid_move(go, side):
#     if go.n_move == 0:
#         possible_moves = [(1, 1), (1, 2), (2, 2)]
#     elif go.n_move == 1:
#         possible_moves = [(i, j) for j in range(5) for i in range(5)
#                           if i != 0 and i != 4 and j != 0 and j != 4 and go.board[i][j] == 0]
#     else:
#         possible_moves = [(i, j) for j in range(5) for i in range(5) if go.valid_place_check(i, j, side)]
#         if len(possible_moves) == 0:
#             possible_moves = [(-1, -1)]
#     return possible_moves


def read_moves():
    with open('n_move.txt', 'r') as f:
        n_move = int(f.readline())
    return n_move


def write_moves(n_move):
    with open('n_move.txt', 'w') as f:
        f.write(str(n_move))


# if __name__ == "__main__":
#     N = 5
#     # read input
#     piece_type, previous_board, board = readInput(N)
#     # load move
#     if os.path.exists('n_move.txt') and np.sum(previous_board) == 0:
#         os.remove('n_move.txt')
#     if os.path.exists('n_move.txt'):
#         n_move = read_moves()
#     else:
#         n_move = 1 if np.sum(board) > 0 else 0
#     # initialize game
#     go = GO(N)
#     go.set_board(piece_type, previous_board, board)
#     go.n_move = n_move
#     # load player
#     player = TDLearner(alpha=0, epsilon=0, max_time=9.0, max_iter=10000)
#     player.load(path='model/')
#     # take action and output
#     action = player.move(go, piece_type, mode='test')
#     writeOutput(action)
#     write_moves(n_move + 2)
