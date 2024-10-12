# %%
import numpy as np
import random

# %%
# Hyperparameters
N = 3  # Size of the Tic-Tac-Toe board

# %%
# calculated constants
winning_combinations = []

# Horizontal combinations
for i in range(N):
    winning_combinations.append([i * N + j for j in range(N)])  # Row i

# Vertical combinations
for j in range(N):
    winning_combinations.append([i * N + j for i in range(N)])  # Column j

# Diagonal combinations
winning_combinations.append([i * (N + 1) for i in range(N)])  # Main diagonal
winning_combinations.append([i * (N - 1)
                            for i in range(1, N + 1)])  # Anti-diagonal

winning_combinations

# %%


def initialize_board():
    return np.zeros(N*N, dtype=int)


# %%
board = initialize_board()
# board

# %%


def isvacant(board, position):
    return board[position] == 0

# %%


def make_move(board, position, player):
    if position < 0 or position >= len(board):
        raise Exception(f"Invalid position {position}")

    if not (isvacant(board, position)):
        raise Exception(f"Not vacant position {position}")

    if player == 1:
        board[position] = 1
    elif player == 2:
        board[position] = 2
    else:
        raise Exception(f"Invalid player {player}")
    return board


def unmove(board, position, player=None):

    if (isvacant(board, position)):
        raise Exception(f"Vacant position {position}")
    board[position] = 0
    return board

# make_move(board, 0, 1)

# %%


def check_winner(board):
    for combination in winning_combinations:
        # if board[combination[0]] == board[combination[1]] == board[combination[2]] != 0:

        if all(board[pos] == board[combination[0]] != 0 for pos in combination):
            return board[combination[0]]  # Return the winning player (1 or 2)

            # Player 1 (x) wins
            # Player 2 (o) wins

    if is_board_full(board):
        return 0  # Draw

    return -1  # game running


# board = [2, 2, 2, 1, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0]
# check_winner(board)

# %%


def is_board_full(board):
    return all(cell != 0 for cell in board)

# %%


def reset_board(board):
    return initialize_board()

# %%


def get_available_moves(board):
    return [i for i in range(N*N) if board[i] == 0]


# %%
def print_board(board):
    mapping = {0: ' ', 1: 'X', 2: 'O'}  # Mapping for display
    for i in range(N):
        # Extract the row from the flattened array
        row = board[i * N:(i + 1) * N]
        print('|'.join(mapping[cell] for cell in row))
        print('-' * (N * 2 - 1))  # Print a separator line


# print_board(board)

# %%


def play_game(agent1=None, agent2=None):
    board = initialize_board()  # Initialize the board as a flattened array
    current_player = 1  # Player 1 starts
    game_over = False

    while not game_over:
        print()
        print_board(board)  # Display the current board

        # Get available moves
        available_moves = get_available_moves(board)
        print(f"Available moves: {available_moves}")

        if current_player == 1:
            agent = agent1
            player_symbol = 'X'
        else:
            agent = agent2
            player_symbol = 'O'

        if agent:  # If there is an agent, it selects a move
            move = agent.select_move(
                board, available_moves)  # Bot selects move
            print(f"Bot ({player_symbol}) selects move: {move}")
        else:  # If no agent, ask for human input
            # Adjust for N size
            move = int(input(
                f"Player {current_player} ({player_symbol}), enter your move (0-{N**2-1}): "))

        if move in available_moves:
            make_move(board, move, current_player)  # Make the move
            winner = check_winner(board)  # Check for a winner

            if winner != -1:  # If the game is not still running
                game_over = True
                print_board(board)  # Display the final board
                if winner == 0:
                    print("It's a draw!")
                else:
                    print(f"Player {winner} wins!")
                return winner
        else:
            print("Invalid move, please try again.")

        # Switch players
        # current_player = 2 if current_player == 1 else 1
        current_player = 3 - current_player


def random_move(board, available_moves):
    # return np.random.choice(available_moves)
    return random.choice(available_moves)

    # Logic for random move


def find_winning_move(board, available_moves, player_number):
    for move in available_moves:
        board[move] = player_number
        winner = check_winner(board)
        board[move] = 0
        if winner == player_number:
            return move
    return -1
# %%
# play_game()

# %%


# %%
