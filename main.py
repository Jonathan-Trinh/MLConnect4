import os
import pygame
import pygame.freetype
import math
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classes.Piece import Piece
from classes.Players import *

pygame.init()
pygame.font.init()
FONT = pygame.font.SysFont('Arial', 24)

BOARD_DIM = (7, 6) #columns, rows
BUFFER_PIXELS = 25
DISPLAY_DIM = (1200, 650)
DISPLAY = pygame.display.set_mode(DISPLAY_DIM)
CLOCK = pygame.time.Clock()

PIECE_IMGS = {-1: pygame.image.load('imgs/red_piece.png'), 1: pygame.image.load('imgs/yellow_piece.png')}
PIECE_DIM = (PIECE_IMGS[-1].get_width(), PIECE_IMGS[-1].get_height())
BOARD_IMG = pygame.image.load('imgs/board.png')

def main(p1='RLbotDDQN', p2='bot', epochs=1000, self_play=False, expr_dir=None, test_paths=None):
    # SETUP & HELPER FUNCTIONS ======================================================
    SCORE_COUNTS = {-1: 0.0, 1: 0.0}
    WIN_RATES =    {-1: 0.0, 1: 0.0}

    def refresh_game():
        global BOARD_ARRAY
        global PIECE_ARRAYS
        global PLACED_PIECES
        global GAME_SEQ
        global RESULT

        BOARD_ARRAY = np.zeros(tuple(reversed(BOARD_DIM)))
        PIECE_ARRAYS = {-1: np.zeros_like(BOARD_ARRAY), 1: np.zeros_like(BOARD_ARRAY)}
        PLACED_PIECES = []
        GAME_SEQ = []
        RESULT = None

    refresh_game()

    def render_scores(names, player):
        p1_name, p2_name = names
        score_x = BUFFER_PIXELS + BOARD_DIM[0] * PIECE_DIM[0] + 50
        score_y = BUFFER_PIXELS
        n_games = SCORE_COUNTS[-1] + SCORE_COUNTS[1]
        DISPLAY.blit(FONT.render(f"P1-{p1_name}: Win rate={WIN_RATES[-1]}", False, (0, 0, 0)), (score_x, score_y))
        DISPLAY.blit(FONT.render(f"P2-{p2_name}: Win rate={WIN_RATES[1]}", False, (0, 0, 0)), (score_x, score_y * 3))
        DISPLAY.blit(FONT.render(f"Games Played: {n_games}", False, (0, 0, 0)), (score_x, score_y * 6))

        # Render RL-specific stats if player is RLBot
        if isinstance(player, RLBot):
            DISPLAY.blit(FONT.render(f"P1 Current Min. Loss: {np.round(player.min_loss_dict['current_min'], 4)}", False, (0, 0, 0)), (score_x, score_y * 9))
            DISPLAY.blit(FONT.render(f"P1 Loss Steps: {player.min_loss_dict['num_steps']}", False, (0, 0, 0)), (score_x, score_y * 11))
            DISPLAY.blit(FONT.render(f"P1 Current Epsilon: {player.epsilon}", False, (0, 0, 0)), (score_x, score_y * 13))

    def render_board():
        DISPLAY.blit(BOARD_IMG, (0 ,0))

    def render_pieces():
        for piece in PLACED_PIECES:
            DISPLAY.blit(PIECE_IMGS[piece.turn], piece.coords)

    def check_connect_4_piece(connect_n: int, piece_array: np.array, placed_coord) -> bool:
        """Determines if the placement of a new piece at @placed_coord results in connect 4."""
        crawl_axes = np.array([
            [1, 0], # vertical
            [0, 1], # horizontal
            [1, 1],
            [-1, 1],
        ])
        def coord_is_inbounds(coord, board) -> bool:
            return (0 <= coord[0] < board.shape[0]) and (0 <= coord[1] < board.shape[1])
        
        axes_sums = []
        for crawl_dir in crawl_axes:
            sum_dir = 1
            search_sides = [True, True] # left, right

            for n in range(connect_n-1):
                left = placed_coord - (n+1)*crawl_dir
                right = placed_coord + (n+1)*crawl_dir
                for i, side in enumerate([left, right]):
                    if search_sides[i] and coord_is_inbounds(side, piece_array):
                        if piece_array[tuple(side)]==0:
                            search_sides[i] = False # cut the chain
                        else:
                            sum_dir += 1
                    else:
                        search_sides[i] = False
                if not any(search_sides):
                    break 
            axes_sums.append(sum_dir)
        
        return any([s>=connect_n for s in axes_sums])

    def place_piece(turn, col):
        """Places piece, updates boards, then assesses board and determines if connect 4 OR draw is found"""
        sum_col = sum(BOARD_ARRAY[:, col])
        is_legal_move = sum_col < BOARD_ARRAY.shape[0]
        if not is_legal_move:
            return f'win-illegal_{str(-turn)}'
        
        if sum_col==0:
            y_row = BOARD_ARRAY.shape[0]-1
        else:
            y_row = BOARD_ARRAY[:, col].argmax()-1

        y_coord = (y_row*PIECE_DIM[0])+BUFFER_PIXELS
        x_coord = col*PIECE_DIM[0]+25
        piece = Piece(turn=turn, coords=(x_coord, y_coord))
        PLACED_PIECES.append(piece)
        BOARD_ARRAY[y_row, col] = 1
        PIECE_ARRAYS[turn][y_row, col] = 1
        GAME_SEQ.append(col)

        """Check win/draw conditions: for each side's PIECE_ARRAY, take the sum of: 
        - all rows, columns, and diagonals
        - if any sum is >= 4, then 'win'
        - if BOARD_ARRAY is full and no 'win', then 'draw'
        """
        connect_4 = check_connect_4_piece(4, PIECE_ARRAYS[turn], placed_coord=(y_row, col))
        board_full = (BOARD_ARRAY==1).all()
        if connect_4:
            return f'win-connect4_{str(turn)}'
        elif board_full:
            return 'draw'
        return None

    def log_result(results_df, first_turn):
        if 'draw' in RESULT:
            SCORE_COUNTS[-1] += 0.5
            SCORE_COUNTS[1] += 0.5
        else:
            win_side = RESULT.split('_')[-1]
            SCORE_COUNTS[int(win_side)] += 1.0
        n_games = sum([SCORE_COUNTS[-1], SCORE_COUNTS[1]])
        WIN_RATES[-1] = round(SCORE_COUNTS[-1]/n_games, 2)
        WIN_RATES[1] = round(SCORE_COUNTS[1]/n_games, 2)

        seq_string = ''.join([str(n) for n in GAME_SEQ])+"_"
        new_result_df = pd.DataFrame({
            'result': RESULT,
            'first_turn': first_turn,
            'seq': seq_string,
            'win_rate_-1': WIN_RATES[-1],
            'win_rate_1': WIN_RATES[1]},
            index=[0]
            )
        return pd.concat([results_df, new_result_df])

    def log_models_and_results(players_dict, results_df, expr_dir, expr_name):
        model_path = f'models/{expr_dir}/{expr_name}/'
        os.makedirs(model_path, exist_ok=True)

        for i, player in players_dict.items():
            # only store current self-play bot
            if isinstance(player, RLBot) and player.turn==-1: 
                player.save_model_and_results(model_path)
            
        results_df.reset_index(drop=True, inplace=True)
        results_df.plot(y=['win_rate_-1', 'win_rate_1'])
        plt.xlabel('Epoch')
        plt.ylabel('Avg. Win Rate')
        plt.savefig(model_path+'avg_win_rates.png')

        results_df.to_csv(model_path+'results_df.csv')

    def log_expr_results(results_dict):
        results_expr = pd.DataFrame(results_dict, index=[0])

        results_all_path = 'models/results_all.csv'
        if os.path.exists(results_all_path):
            results_all = pd.read_csv(results_all_path, index_col=0)
            results_all = pd.concat([results_all, results_expr])
        else:
            results_all = results_expr
        results_all.to_csv(results_all_path)

    def build_expr_dict(**kwargs):
        expr_dict = {'expr_name': ''}
        for var, var_val in kwargs.items():
            if var_val not in ['', None]:
                expr_dict[var] = var_val
                expr_dict['expr_name'] += f'_{var}-{var_val}'
        return expr_dict
    

    # MAIN GAME LOOP ==============================================================
    global RESULT
    is_test = test_paths is not None
    results_df = pd.DataFrame()
    player_classes = {'player': Human, 'bot': Bot, 'RLbot': RLBot, 'RLbotDDQN': RLBotDDQN}
    players = [p1, p2]
    p1_turn = -1
    if self_play:
        players = {-1: p1, 1: p2} # RL bot to be trained on self-play is always '-1'
    else:
        players = {k: player_classes[players[v]](name=n, turn=k, test=is_test) \
                for k, v, n in zip([p1_turn, -p1_turn], [0, 1], players)}
    first_turn = -1
    curr_turn = -1
    curr_player = players[curr_turn]
    while True:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
            # take turn
            if event.type==pygame.MOUSEBUTTONDOWN and curr_player.name=='player':
                col = math.floor((event.pos[0]-BUFFER_PIXELS)/PIECE_DIM[0])
                if 0 <= col < 7:
                    if BOARD_ARRAY[0, col]==0:
                        RESULT = place_piece(curr_turn, col)
                        curr_turn *= -1; curr_player = players[curr_turn]
        if 'bot' in curr_player.name:
            is_rl_bot = 'RLbot' in curr_player.name
            passed_state = PIECE_ARRAYS if is_rl_bot else BOARD_ARRAY
            # Determine model weights based on simulation parameters
            if is_test and (len(GAME_SEQ) < 2) and is_rl_bot: # first moves
                curr_player.load_model(test_paths[curr_player.turn])
            if self_play and all('RLbot' in p for p in [p1, p2]): # copy other RLbot's weights
                curr_player.model.load_state_dict(players[-curr_turn].model.state_dict())
            # get & make move
            col = curr_player.move(passed_state)
            RESULT = place_piece(curr_turn, col)
            if is_rl_bot and not is_test:
                curr_player.train(PIECE_ARRAYS, RESULT)
            curr_turn *= -1; curr_player = players[curr_turn]

        DISPLAY.fill("white")
        render_pieces()
        render_board()
        render_scores(names=[p if not isinstance(p, Player) else p.name for p in [p1, p2]], player=players[-1])
        pygame.display.flip()
        CLOCK.tick(60)

        if RESULT is not None:
            results_df = log_result(results_df, first_turn)
            refresh_game()
            first_turn *= -1; curr_player = players[first_turn]
            if not is_test and isinstance(curr_player, RLBot) and curr_player.turn == -1 and curr_player.stop_training:
                break

        if results_df.shape[0]==epochs:
            break

    expr_dict = build_expr_dict(
        date=datetime.datetime.today().strftime('%m-%d-%Y-%H-%M'),
        p1=p1, p2=p2,
        self_play=self_play,
        expr_ext=expr_dir,
        epochs=epochs,
        p1_win_rate=WIN_RATES[players[p1_turn].turn],
        p2_win_rate=WIN_RATES[-p1_turn],
        is_test=is_test
    )
    log_expr_results(expr_dict)
    log_models_and_results(players, results_df, expr_dir, expr_dict['expr_name'])

if __name__=="__main__":
    pairings = [
        ['bot', 'bot', False],
        ['RLbot', 'bot', False],
        ['RLbot', 'RLbot', False],
        ['RLbot', 'RLbot', True],
    ]

    for a, b, play_type in pairings:
        main(
            p1=a,
            p2=b,
            epochs=1000,
            expr_dir='base_sims',
            self_play=play_type,
        )