#include <stdint.h>
#include <stdbool.h>

typedef enum error {
    SUCCESS = 0, 
} Error;

typedef enum piece {
    EMPTY = 0,
    PAWN,
    CANNON,
    ROOK,
    KNIGHT,
    ELEPHANT,
    GUARD,
    KING,
} Piece;

typedef struct location {
    int8_t row;
    int8_t col;
} Location;

static const Location UNIT_DIRECTIONS[4] = { {0,1}, {0,-1},  {1, 0}, {-1, 0}};

static inline Location translate_location(Location loc, int8_t direction, int8_t steps) {
    return (Location){loc.row + UNIT_DIRECTIONS[direction].row * steps, loc.col + UNIT_DIRECTIONS[direction].col * steps};
}
 
inline int8_t get_piece(const int8_t* board, Location loc) {
    return board[loc.col + (loc.row * 9)];
}

inline int8_t get_player(int8_t piece) {
    return piece < 0 ? -1 : piece & 1;
}

inline bool is_within_bounds(Location loc) {
    return (loc.row >= 0) && (loc.row < 10) && (loc.col >= 0) && (loc.col < 9);
}

inline bool is_across_river(int8_t player, Location loc) {
    return (2 * loc.row - 9) * player > 0;
}

inline bool is_in_palace(int8_t player, Location loc) {
    return ((2 * loc.row - 9) * player < -4) && (loc.col > 2) && (loc.col < 6);
}

inline bool is_valid_move(const int8_t* board, int8_t player, Location move) {
    return is_within_bounds(move) && get_player(get_piece(board, move)) != player;
}

int8_t pawn_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    moves[0] = (Location){loc.row + player, loc.col};
    n_moves += is_valid_move(board, player, moves[0]);
    if(is_across_river(player, loc)) {
        moves[1] = (Location){loc.row, loc.col + 1};
        n_moves += is_valid_move(board, player, moves[1]);
        moves[2] = (Location){loc.row, loc.col - 1};
        n_moves += is_valid_move(board, player, moves[2]);
    }
    return n_moves;
}

int8_t cannon_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir, i;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1);
        for(i = 2; is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY; ++i) {
            moves[++n_moves] = translate_location(loc, dir, i);
        }
        ++i;
        moves[n_moves] =  translate_location(loc, dir, i);
        for(++i; is_within_bounds(moves[n_moves]); ++i) {
            if(get_player(get_piece(board, moves[n_moves])) == -player) {
                ++n_moves;
                break;
            }
            moves[n_moves] =  translate_location(loc, dir, i);
        }
    }
    return n_moves;
}

int8_t rook_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir, i;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1);
        for(i = 2; is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY; ++i) {
            moves[++n_moves] = translate_location(loc, dir, i);
        }
        n_moves += is_within_bounds(moves[n_moves]) && get_player(get_piece(board, moves[n_moves]) == -player);
    }
    return n_moves;
}

/*


def knight_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = []
    for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if (is_within_bounds((row + d_row, col + d_col))
                and board[row + d_row, col + d_col] == EMPTY):
            moves.extend([
                move for move in (
                    (row + 2 * d_row + d_col, col + 2 * d_col + d_row),
                    (row + 2 * d_row - d_col, col + 2 * d_col - d_row),
                ) if is_valid_move(board, player, move)
            ])
    return moves


def elephant_moves(board, location):
    row, col = location
    player = get_player(board[location])
    return [
        move for move in ((row + 2, col + 2), (row + 2, col - 2),
                          (row - 2, col + 2), (row - 2, col - 2))
        if is_valid_move(board, player, move)
        and not is_across_river(player, move)
        and board[(move[0] + row) // 2, (move[1] + col) // 2] == EMPTY
    ]


def guard_moves(board, location):
    row, col = location
    player = get_player(board[location])
    return [
        move for move in ((row + 1, col + 1), (row + 1, col - 1),
                          (row - 1, col + 1), (row - 1, col - 1))
        if is_valid_move(board, player, move) and is_in_palace(player, move)
    ]


def king_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = [
        move for move in ((row, col + 1), (row, col - 1), (row + 1, col),
                          (row - 1, col))
        if is_valid_move(board, player, move) and is_in_palace(player, move)
    ]
    col += player
    while is_within_bounds((row, col)):
        if board[row, col] != EMPTY:
            if enemy(board[(row, col)]) == board[location]:
                moves.append((row, col))
            break
        col += player
    return moves

def get_banned_move(move_history):
    if len(move_history) < 3:
        return None
    opponent_location, opponent_move = move_history[-1]
    if (opponent_move, opponent_location) == move_history[-3]:
        move, location = move_history[-2]
        return (location, move)


def move_piece(board, location, move):
    board = np.copy(board)
    reward = rewards_lookup[board[move]]
    board[move] = board[location]
    board[location] = EMPTY
    return board, reward
*/

#include <stdio.h>

int main() {
    int8_t board[90];
    int8_t i = 0;
    int8_t n_moves;
    for(i = 0; i < 90; ++i) {
        board[i] = 0;
    }
    //board[3] = 1;
    board[6] = -1;
    Location moves[100];

    board[0] = 1;
    n_moves = rook_moves(moves, board, (Location){0,0});
 
    for(i = 0; i < n_moves; ++i) {
        printf("(%d,%d)\r\n", moves[i].row, moves[i].col);
    }
    return 0;
}
