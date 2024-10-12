# %%
# pip install ultraimport

# %% [markdown]
# ## supress print

# %%
from tqdm import tqdm
import random
import sys
import os
import ultraimport
import builtins

# Step 1: Store the original print function
original_print = builtins.print


def suppress_print():
    """Temporarily suppress print statements."""
    builtins.print = lambda *args, **kwargs: None  # Override print with a no-op


def restore_print():
    """Restore the original print function."""
    builtins.print = original_print  # Restore to the saved original print function


# %%

# %%
# os.getcwd()

# %%
ultraimport('../scripts/helper.py', '*', add_to_ns=globals())

# %%
board = initialize_board()
board

# %% [markdown]
# ## Trimmed heuristics

# %% [markdown]
# ### center control

# %%


def heuristic_center_control(board):
    return 3 if board[4] == 'X' else -3 if board[4] == 'O' else 0

# %% [markdown]
# ### Winnig lines

# %%


def winning_lines(board):
    lines = []
    for combo in winning_combinations:
        lines.append([board[i] for i in combo])
    return lines


def heuristic_one_move_win(board):
    lines = winning_lines(board)
    x_winning_lines = sum(1 for line in lines if line.count(
        1) == 2 and line.count(0) == 1)
    o_winning_lines = sum(1 for line in lines if line.count(
        2) == 2 and line.count(0) == 1)

    return x_winning_lines * 5 - o_winning_lines * 5


def heuristic_two_move_win(board):
    lines = winning_lines(board)
    x_winning_lines = sum(1 for line in lines if line.count(
        1) > 0 and line.count(2) == 0)
    o_winning_lines = sum(1 for line in lines if line.count(
        2) > 0 and line.count(1) == 0)

    return x_winning_lines * 5 - o_winning_lines * 5


# %% [markdown]
# ## Minimax

# %%


def minimax_score(board, eval_strat=None):
    result = check_winner(board)

    if result == 1:  # Player 1 (X) wins
        return 10
    elif result == 2:  # Player 2 (O) wins
        return -10
    elif result == -1:  # Game is ongoing
        score = 0

        # Call the appropriate heuristic function based on strategy
        if eval_strat == 'two_move':
            score += heuristic_one_move_win(board)
        elif eval_strat == 'center_control':
            score += heuristic_center_control(board)
        elif eval_strat == 'one_move':
            score += heuristic_two_move_win(board)
        # else:
        #     raise Exception(f"Invalid eval strategy {eval_strat}")
        return score

    return 0  # Neutral score for draw or ongoing


# %%


def minimax(board, depth, alpha, beta, player_number, eval_strat=None):
    # Check for terminal state first
    score = minimax_score(board)
    if score != 0 or depth == 0:  # If terminal state or depth is 0
        return None, score  # Return score for evaluation

    available_moves = get_available_moves(board)

    best_move = None
    if player_number == 1:
        best_score = -sys.maxsize - 1  # Initialize to negative infinity
        for move in available_moves:
            board[move] = player_number
            _, score = minimax(board, depth - 1, alpha,
                               beta, 2)  # Switch to player 2
            board[move] = 0  # Undo the move
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if beta <= alpha:  # Alpha-beta pruning
                break
    else:  # Minimizing for player 2
        best_score = sys.maxsize  # Initialize to positive infinity
        for move in available_moves:
            board[move] = player_number
            _, score = minimax(board, depth - 1, alpha,
                               beta, 1)  # Switch to player 1
            board[move] = 0  # Undo the move
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)
            if beta <= alpha:  # Alpha-beta pruning
                break

    return best_move, best_score


# %% [markdown]
# ## Agent

# %%
class Agent:
    def __init__(self, player_number, strategy='random', depth=None, eval_strat=None, ):
        if strategy.startswith('minmax') and strategy != 'minmax':
            strategy, eval_strat, depth = strategy.split(' ')
            depth = int(depth)

        self.player_number = player_number
        self.strategy = strategy
        self.depth = depth
        self.eval_strat = eval_strat
        self.trainable = False

    def select_move(self, board, available_moves=None):
        if available_moves is None:
            available_moves = get_available_moves(board)
        if self.strategy == 'random':
            return random_move(board, available_moves)
        elif self.strategy == 'greedy':
            return self.greedy_move(board, available_moves, self.player_number)
        elif self.strategy == 'block':
            return self.greedy_move(board, available_moves, self.player_number)
        elif self.strategy == 'center_corner':
            return self.center_corner_move(board, available_moves)
        elif self.strategy == 'greedy_center':
            return self.greedy_center_move(board, available_moves, self.player_number)
        elif self.strategy == 'minmax':
            depth = self.depth
            if depth is None:
                depth = len(available_moves)

            move, _ = minimax(board, depth, -99, 99,
                              self.player_number, self.eval_strat)
            return move

        else:
            raise Exception(f'Invalid strategy {strategy}')
        # Add more strategies as needed

    def block_move(self, board, available_moves, player_number):
        # try to block opponent win
        other_player = 3 - player_number
        move = find_winning_move(board, available_moves, other_player)
        if move != -1:
            return move
        else:
            return random_move(board, available_moves)

    def greedy_move(self, board, available_moves, player_number):
        # Try to win in single turn
        move = find_winning_move(board, available_moves, player_number)
        if move != -1:
            return move

        # return self.block_move(board, available_moves, player_number)
        return random_move(board, available_moves)

    def center_corner_move(self, board, available_moves):
        # Flattened board indices for center, corners, and edges
        # For a 3x3 board, center is index 4 (0-based)
        center = [(N**2 - 1) // 2]
        # Top-left, top-right, bottom-left, bottom-right
        corners = [0, N-1, N**2-N, N**2-1]
        # edges = [i for i in range(N**2) if i not in center + corners]  # All non-corner, non-center positions

        # First, check if the center is available
        if board[center[0]] == 0:  # Assuming 0 means empty
            return center[0]

        # If center is not available, check for available corners
        available_corners = [i for i in corners if board[i] == 0]
        if available_corners:
            # Randomly select one if multiple are available
            return random_move(board, available_corners)

        # # If neither center nor corners are available, move to an edge
        # available_edges = [i for i in edges if board[i] == 0]
        # if available_edges:
        #     return random.choice(available_edges)  # Randomly select one if multiple are available

        # If no moves are available (shouldn't happen if the game is valid), return None
        return random_move(board, available_moves)

    def greedy_center_move(self, board, available_moves, player_number):
        # Try to win in single turn
        move = find_winning_move(board, available_moves, player_number)
        if move != -1:
            return move

        other_player = 3 - player_number
        move = find_winning_move(board, available_moves, other_player)
        if move != -1:
            return move

        return self.center_corner_move(board, available_moves)


# %%
# %%
# %%

# Define the board size
N = 3  # For a standard Tic-Tac-Toe board, N = 3


def select_center_corner_edge_move(board):
    # Flattened board indices for center, corners, and edges
    center = [(N**2 - 1) // 2]  # For a 3x3 board, center is index 4 (0-based)
    # Top-left, top-right, bottom-left, bottom-right
    corners = [0, N-1, N**2-N, N**2-1]
    # All non-corner, non-center positions
    edges = [i for i in range(N**2) if i not in center + corners]

    # First, check if the center is available
    if board[center[0]] == 0:  # Assuming 0 means empty
        return center[0]

    # If center is not available, check for available corners
    available_corners = [i for i in corners if board[i] == 0]
    if available_corners:
        # Randomly select one if multiple are available
        return random.choice(available_corners)

    # If neither center nor corners are available, move to an edge
    available_edges = [i for i in edges if board[i] == 0]
    if available_edges:
        # Randomly select one if multiple are available
        return random.choice(available_edges)

    # If no moves are available (shouldn't happen if the game is valid), return None
    return None


# Example usage
# 0 represents an empty position, 1 for player 1, and 2 for player 2
board = [1, 0, 0, 2, 0, 2, 0, 1, 1]  # Example current board state
# move = select_center_corner_edge_move(board)
# print(f"Selected move: {move}")


# %% [markdown]
# ## Round robin Tournament

# %%

# %%

def round_robin_tournament(strategies, rounds=1, agents=None, verbose=True, self_games=True, tqdm_flag=True, random_seed=42):
    # Step 1: Create a list to hold results
    results = {strategy: {"wins": 0, "losses": 0, "draws": 0}
               for strategy in strategies}

    num_strategies = len(strategies)
    win_grid = np.zeros((num_strategies, num_strategies))
    strategy_indices = {strategy: idx for idx, strategy in enumerate(
        strategies)}  # Mapping strategy to grid index

    # Step 2: Create agents for each strategy
    if agents is not None:
        agents_1 = dict(zip(strategies, agents))
        agents_2 = dict(zip(strategies, agents))
        # agents_2 = agents

    else:
        agents_1 = {strategy: Agent(player_number=1, strategy=strategy)
                    for strategy in strategies}
        agents_2 = {strategy: Agent(player_number=2, strategy=strategy)
                    for strategy in strategies}

    # Step 3: Play games among agents
    total_matches = len(strategies) * (len(strategies) -
                                       1 + int(self_games)) * rounds

    k = 1
    if verbose:
        suppress_print()
    with tqdm(total=total_matches, desc='Playing matches') as pbar:
        for round in range(rounds):

            for strategy1 in strategies:
                print()
                for strategy2 in strategies:
                    # Skip if both strategies are the same
                    if not (self_games) and strategy1 == strategy2:

                        continue

                    agent1 = agents_1[strategy1]
                    agent2 = agents_2[strategy2]
                    # Step 4: Create two agents for the current match

                    # for match in range(2):
                    if True:

                        # # Temporarily set player numbers for the match
                        # agent1.player_number = match
                        # agent2.player_number = 3 - match
                        if random_seed is not None:
                            # Initialize before the match
                            random.seed(random_seed)
                        # if match == 1:
                        #     # swap(strategy1, strategy2)
                        #     strategy1, strategy2 =  strategy2, strategy1

                        suppress_print()
                        # Play game and get the result
                        winner = play_game(agent1=agent1, agent2=agent2)

                        if verbose:
                            restore_print()
                        # Step 5: Update results based on the winner
                        if winner == 0:  # Draw
                            results[strategy1]["draws"] += 1
                            results[strategy2]["draws"] += 1
                        elif winner == 1:  # agent1 wins
                            results[strategy1]["wins"] += 1
                            results[strategy2]["losses"] += 1
                        elif winner == 2:  # agent2 wins
                            results[strategy2]["wins"] += 1
                            results[strategy1]["losses"] += 1
                        else:
                            print(winner, "wom")

                        # Update win grid
                        idx1 = strategy_indices[strategy1]
                        idx2 = strategy_indices[strategy2]

                        if winner == 1:
                            # Increment win count for strategy1 against strategy2
                            win_grid[idx1, idx2] += 1
                        elif winner == 2:
                            # Increment win count for strategy2 against strategy1
                            win_grid[idx2, idx1] += 1

                        pbar.update(1)

                        # print(f"{k} Player 1 : {strategy1}, Player 2 : {strategy2} result : {winner}")
                        print(
                            f"Match {k} {strategy1} VS {strategy2} : result : {winner}")

                        k += 1

    # if verbose:
    restore_print()

    # Step 6: Convert win counts in the grid to percentages
    # for i in range(num_strategies):
    #     for j in range(num_strategies):
    #         if i != j:  # Skip self-matches
    #             total_games = rounds  # Number of rounds played between strategy i and strategy j
    #             win_grid[i, j] = (win_grid[i, j] / total_games) * 100  # Convert win counts to percentages

    # Step 7: Return the results and the win percentage grid
    return results, win_grid

    # return results

# %%


def single_match(strategy1, strategy2, agent1=None, agent2=None):
    if agent1 is None:
        agent1 = Agent(player_number=1, strategy=strategy1)
    if agent2 is None:
        agent2 = Agent(player_number=2, strategy=strategy2)
    winner = play_game(agent1=agent1, agent2=agent2)


# %%
# single_match('minmax', 'greedy')
# %%
board = np.array([1, 1, 2, 0, 0, 2, 0, 0, 0])
# print_board(board)

# %%
# minimax(board, 5,)

# %%
# single_match('minmax','greedy_center')
# single_match('random','random')
# single_match('greedy_center','minmax')

# %%


# Example usage
# strategies = ['random', 'greedy', 'block', 'center_corner']
strategies = ['random', 'greedy', 'block', 'center_corner', 'greedy_center',
              'minmax one_move 1',
              'minmax one_move 2',
              'minmax one_move 3',
              'minmax one_move 4',
              'minmax one_move 4',
              'minmax one_move 5',
              'minmax one_move 6',
              'minmax one_move 7',
              'minmax']
# strategies = ['random', 'greedy', 'block', 'center', 'corner', 'edge', 'alpha-beta']
rounds = 1
VERBOSE = False
SELF_GAMES = False
# tournament_results, win_grid = round_robin_tournament(strategies, rounds, verbose= VERBOSE, self_games = SELF_GAMES)

# Display results
# print()

# for strategy, result in tournament_results.items():
#     print(f"{strategy}: Wins: {result['wins']}, Losses: {result['losses']}, Draws: {result['draws']}")


# %%


# %% [markdown]
# ## Display results

# %%
# tournament_results

# %%
# win_grid

# %%
if __name__ == "__main__":

    from tabulate import tabulate

    # Assume strategies is a list of strategy names
    # Assume tournament_results is a dictionary holding results for each strategy
    total_matches = 2 * len(strategies) * (len(strategies) - 1) * rounds
    matches_per_strat = total_matches/len(strategies)
    # Organize results into a list of lists
    results = [
        [strategy, info['wins'], info['losses'], info['draws'],  # Absolute counts
         f"{(info['wins'] / matches_per_strat) * 100:.2f}%",  # Win %
         f"{(info['losses'] / matches_per_strat) * 100:.2f}%",  # Loss %
         f"{(info['draws'] / matches_per_strat) * 100:.2f}%"]  # Draw %
        for strategy, info in tournament_results.items()
    ]

    # Create a table string
    table_string = tabulate(results, headers=[
                            'Strategy', 'Wins', 'Losses', 'Draws', 'Win %', 'Loss %', 'Draw %'], tablefmt='grid')

    # Print the table
    print(table_string)

    # Optionally save the table to a .txt file
    # with open('tournament_results.txt', 'w') as file:
    #     file.write(table_string)

    # %%
    # Assume win_grid is the MxM numpy array holding win percentages between strategies
    # win_grid = np.array([[0, 33.33, 66.67], [66.67, 0, 33.33], [33.33, 66.67, 0]])  # Example data

    win_grid = win_grid
    # Create a list of lists with strategies as rows and columns, including the win percentages
    grid_results = [
        ([strategies[i]] +
         [f"{win_grid[i, j] * 100 / 2:.2f}%" for j in range(len(strategies))])
        for i in range(len(strategies))
    ]

    # Add the strategy names as headers (the first row/column)
    headers = ["Strategy"] + strategies

    # Create the table string using tabulate
    table_string = tabulate(grid_results, headers=headers, tablefmt='grid')

    # Print the table
    print(table_string)

    # %%
    # Assume win_grid is the MxM numpy array holding win percentages between strategies
    # win_grid = np.array([[0, 33.33, 66.67], [66.67, 0, 33.33], [33.33, 66.67, 0]])  # Example data

    win_grid = win_grid
    # Create a list of lists with strategies as rows and columns, including the win percentages
    grid_results = [
        ([strategies[i]] +
         [f"{win_grid[i, j] * 100 / 2:.2f}%" for j in range(len(strategies))])
        for i in range(len(strategies))
    ]

    # Add the strategy names as headers (the first row/column)
    headers = ["Strategy"] + strategies

    # Create the table string using tabulate
    table_string = tabulate(grid_results, headers=headers, tablefmt='grid')

    # Print the table
    print(table_string)
