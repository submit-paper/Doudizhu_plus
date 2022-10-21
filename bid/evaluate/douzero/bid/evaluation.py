import multiprocessing as mp
import pickle

from douzero.env.game import GameEnv

num_landlord_wins = mp.Value('i', 0)
num_farmer_wins = mp.Value('i', 0)
num_landlord_scores = mp.Value('i', 0)
num_farmer_scores = mp.Value('i', 0)

player1_win_score = mp.Value('i', 0)
player1_win_num = mp.Value('i', 0)

player2_win_score = mp.Value('i', 0)
player2_win_num = mp.Value('i', 0)

player3_win_score = mp.Value('i', 0)
player3_win_num = mp.Value('i', 0)

def load_card_play_models(card_play_model_path_dict):
    players = {}

    for position in ['landlord', 'landlord_up', 'landlord_down']:
        if card_play_model_path_dict[position] == 'rlcard':
            from .rlcard_agent import RLCardAgent
            players[position] = RLCardAgent(position)
        elif card_play_model_path_dict[position] == 'random':
            from .random_agent import RandomAgent
            players[position] = RandomAgent()
        elif 'baseline' in card_play_model_path_dict[position]:
            from .baseline_agent import BaseAgent
            players[position] = BaseAgent(position, card_play_model_path_dict[position])
        else:
            from .deep_agent import DeepAgent
            players[position] = DeepAgent(position, card_play_model_path_dict[position])
    return players

def mp_simulate(card_play_data_list, card_play_model_path_dict, paras):
    from .bid_agent import BidAgent
    bid = BidAgent('baselines/douzero_ADP/bid.pkl')
    gate0, gate1, gate2 = paras
    
    players = load_card_play_models(card_play_model_path_dict)
    results = []
    env = GameEnv(players, bid, gate0, gate1, gate2)
    for idx, card_play_data in enumerate(card_play_data_list):
        env.card_play_init(card_play_data)
        while not env.game_over:
            env.step()
        env.reset()

    with num_landlord_wins.get_lock():
        num_landlord_wins.value += env.num_wins['landlord']

    with num_farmer_wins.get_lock():
        num_farmer_wins.value += env.num_wins['farmer']

    with num_landlord_scores.get_lock():
        num_landlord_scores.value += env.num_scores['landlord']

    with num_farmer_scores.get_lock():
        num_farmer_scores.value += env.num_scores['farmer']
        
    
    with player1_win_score.get_lock():
        player1_win_score.value += env.player_score[0]
        
    with player1_win_num.get_lock():
        player1_win_num.value += env.player_win[0]
        
    with player2_win_score.get_lock():
        player2_win_score.value += env.player_score[1]
        
    with player2_win_num.get_lock():
        player2_win_num.value += env.player_win[1]
    
    with player3_win_score.get_lock():
        player3_win_score.value += env.player_score[2]
        
    with player3_win_num.get_lock():
        player3_win_num.value += env.player_win[2]

def data_allocation_per_worker(card_play_data_list, num_workers):
    card_play_data_list_each_worker = [[] for k in range(num_workers)]
    for idx, data in enumerate(card_play_data_list):
        card_play_data_list_each_worker[idx % num_workers].append(data)

    return card_play_data_list_each_worker

def evaluate(landlord, landlord_up, landlord_down, eval_data, num_workers, paras):

    global num_landlord_wins
    global num_farmer_wins
    global num_landlord_scores
    global num_farmer_scores
    with open(eval_data, 'rb') as f:
        card_play_data_list = pickle.load(f)

    card_play_data_list_each_worker = data_allocation_per_worker(
        card_play_data_list, num_workers)
    del card_play_data_list

    card_play_model_path_dict = {
        'landlord': landlord,
        'landlord_up': landlord_up,
        'landlord_down': landlord_down}

    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(mp_simulate,
                                args=(card_play_data,
                                      card_play_model_path_dict, paras))
               for card_play_data in card_play_data_list_each_worker]

    results = [p.get() for p in results]
    # num_total_wins = num_landlord_wins.value + num_farmer_wins.value
    landlord_wins = num_landlord_wins.value
    farmer_wins = num_farmer_wins.value
    landlord_scores = num_landlord_scores.value
    farmer_scores = num_farmer_scores.value
    num_landlord_wins = mp.Value('i', 0)
    num_farmer_wins = mp.Value('i', 0)
    num_landlord_scores = mp.Value('i', 0)
    num_farmer_scores = mp.Value('i', 0)
    print('paras: ', paras)
    print('WP results:')
    print('landlord : Farmers - {} : {}'.format(landlord_wins, farmer_wins))
    print('ADP results:')
    print('landlord : Farmers - {} : {}'.format(landlord_scores, 2 * farmer_scores))
    print('Player 1 win: {}, win score: {}'.format( player1_win_num.value, player1_win_score.value))
    print('Player 2 win: {}, win score: {}'.format( player2_win_num.value, player2_win_score.value))
    print('Player 3 win: {}, win score: {}'.format( player3_win_num.value, player3_win_score.value))
    return landlord_wins
