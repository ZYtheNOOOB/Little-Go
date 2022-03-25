import collections
import time

from tqdm import tqdm
from test import visualize_board

import go as g
from copy import deepcopy

from uct_player import MCTSPlayer
from pattern_learner2 import TDLearner
import numpy as np
import random


def run(agent1, agent2, verbose=False, p=1):
    print('')
    print('##### Test Run #####')
    print('Agent:', p, 'MCTS:', 3 - p)
    # game
    go = g.PyGO()
    piece_type = 1

    # play
    while 1:
        print('')
        if piece_type == 1:
            action = agent1.move(go, piece_type)
        elif piece_type == 2:
            action = agent2.move(go, piece_type)
        else:
            raise ValueError('Invalid piece type')

        if action != 0:
            # If invalid input, continue the loop. Else it places a chess on the board.
            if go.place_chess(action[0], action[1], piece_type):
                go.remove_died_pieces(3 - piece_type)
            else:
                print('Invalid placement at', action)
                result = 3 - piece_type
                break

        value, _ = agent3.compute_state_value(go.board)
        plyr = "MCTS" if piece_type != p else "Agent"
        print('value:', value)
        print('step:', go.n_move + 1, 'move:', plyr, 'at', action)
        visualize_board(go.board)

        go.n_move += 1
        piece_type = 3 - piece_type

        if go.game_end(piece_type, action):
            result = go.judge_winner()
            break

        if action == 0:
            go.place_pass()

    print('game end')
    visualize_board(go.board)
    print('winner', result)
    return reward[result]


def test(agent, mcts, episode=10, verbose=False):
    mcts_plyr = MCTSPlayer(side=2, iter=mcts, time=float('inf'), test_agent=True)
    wins_1 = []
    for _ in range(episode):
        r = run(agent1=agent, agent2=mcts_plyr, verbose=verbose, p=1)
        wins_1.append(r == 1)

    mcts_plyr = MCTSPlayer(side=1, iter=mcts, time=float('inf'), test_agent=True)
    wins_2 = []
    for _ in range(episode):
        r = run(agent1=mcts_plyr, agent2=agent, verbose=verbose, p=2)
        wins_2.append(r == 0)
    return np.average(wins_1), np.average(wins_2)


if __name__ == '__main__':
    # np.random.seed(0)
    # random.seed(0)
    verbose = False
    reward = TDLearner.reward

    # hyperparams
    lr = 0.05

    # black
    agent1 = TDLearner(alpha=lr, epsilon=.1, max_time=8.0, max_iter=20000)
    # white
    agent2 = TDLearner(alpha=lr, epsilon=.1, max_time=8.0, max_iter=20000)
    # agent3 = TDLearner(alpha=lr, epsilon=.1, max_time=9.0)

    agent1.load(path='model2/')
    agent2.load(path='model2/')
    # agent3.load(path='model2/')

    # wr_1, wr_2 = test(agent1, mcts=10000, episode=25)
    # print('Test against pure MCTS %d:' % 10000)
    # print('WR as black:', wr_1)
    # print('WR as white:', wr_2)
    # assert 0

    # train
    episode = 1000000
    best_iter = 20000
    for i in tqdm(range(episode)):
        # game
        go = g.PyGO()
        result = -1
        piece_type = 1

        agent1.TD_history.clear()
        agent2.TD_history.clear()

        # play
        while 1:
            if piece_type == 1:
                action = agent1.move(go, piece_type, mode='train')
            elif piece_type == 2:
                action = agent2.move(go, piece_type, mode='train')
            else:
                raise ValueError('Invalid piece type')

            if action != 0:
                # If invalid input, continue the loop. Else it places a chess on the board.
                if go.place_chess(action[0], action[1], piece_type):
                    go.remove_died_pieces(3 - piece_type)
                else:
                    print('Invalid placement at', action)
                    visualize_board(go.board)
                    raise ValueError('invalid move')

            go.n_move += 1
            piece_type = 3 - piece_type

            if go.game_end(piece_type, action):
                result = go.judge_winner()
                r = reward[result]
                if piece_type == 1:
                    agent1.learn(r=r)
                if piece_type == 2:
                    agent2.learn(r=r)
                break
            else:
                if piece_type == 1:
                    agent1.learn(r=None)
                if piece_type == 2:
                    agent2.learn(r=None)

            if action == 0:
                go.place_pass()

        # test
        if i % 1000 == 0:
            agent1.save(path='model2/')
            print('')
            print('KL 1x1:', np.average([d for k, d in agent1.kl.items() if len(k) == 1]))
            print('KL 2x2:', np.average([d for k, d in agent1.kl.items() if len(k) == 4]))
            print('KL 3x3:', np.average([d for k, d in agent1.kl.items() if len(k) == 9]))
            print('KL 5x5:', np.average([d for k, d in agent1.kl.items() if len(k) == 25]))
            agent1.kl = {}
            agent2.kl = {}
            # if i % 500000 == 0:
            #     wr_1, wr_2 = test(agent1, mcts=best_iter, episode=5)
            #     print('')
            #     print('Test against pure MCTS %d:' % best_iter)
            #     print('WR as black:', wr_1)
            #     print('WR as white:', wr_2)
            #     if wr_1 >= 0.6 and wr_2 >= 0.6:
            #         best_iter += 5000

