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

inline Location translate_location(Location loc, int8_t direction, int8_t forward_steps, int8_t side_steps) {
    switch (direction & 0x3) {
        case 0:
            return (Location){loc.row + forward_steps, loc.col + side_steps};
        case 1:
            return (Location){loc.row + side_steps, loc.col - forward_steps};
        case 2:
            return (Location){loc.row - forward_steps, loc.col - side_steps};
        default: 
            return (Location){loc.row - side_steps, loc.col + forward_steps};
    }
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
        moves[n_moves] = translate_location(loc, dir, 1, 0);
        for(i = 2; is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY; ++i) {
            moves[++n_moves] = translate_location(loc, dir, i, 0);
        }
        ++i;
        moves[n_moves] =  translate_location(loc, dir, i, 0);
        for(++i; is_within_bounds(moves[n_moves]); ++i) {
            if(get_player(get_piece(board, moves[n_moves])) == -player) {
                ++n_moves;
                break;
            }
            moves[n_moves] =  translate_location(loc, dir, i, 0);
        }
    }
    return n_moves;
}

int8_t rook_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir, i;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1, 0);
        for(i = 2; is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY; ++i) {
            moves[++n_moves] = translate_location(loc, dir, i, 0);
        }
        n_moves += is_valid_move(board, player, moves[n_moves]);
    }
    return n_moves;
}

int8_t knight_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1, 0);
        if(is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY) {
            moves[n_moves] = translate_location(loc, dir, 2, 1);
            n_moves += is_valid_move(board, player,  moves[n_moves]);
            moves[n_moves] = translate_location(loc, dir, 2, -1);
            n_moves += is_valid_move(board, player,  moves[n_moves]);
        }
    }
    return n_moves;
}

int8_t elephant_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1, 1);
        if(is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY) {
            moves[n_moves] = translate_location(loc, dir, 2, 2);
            n_moves += is_valid_move(board, player,  moves[n_moves]) && !is_across_river(player, moves[n_moves]);
        }
    }
    return n_moves;
}


int8_t guard_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1, 1);
        n_moves += is_valid_move(board, player,  moves[n_moves]) && is_in_palace(player, moves[n_moves]);
    }
    return n_moves;
}


int8_t king_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t n_moves = 0;
    int8_t player = get_player(get_piece(board, loc));
    int8_t dir, i;
    for(dir = 0; dir < 4; ++dir) {
        moves[n_moves] = translate_location(loc, dir, 1, 0);
        n_moves += is_valid_move(board, player,  moves[n_moves]) && is_in_palace(player, moves[n_moves]);
    }
    moves[n_moves] = (Location){loc.row + player, loc.col};
    for(i = 2; is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) == EMPTY; ++i) {
        moves[n_moves] = (Location){loc.row + (i * player), loc.col};
    }
    n_moves += is_within_bounds(moves[n_moves]) && get_piece(board, moves[n_moves]) * player == -KING;
    return n_moves;
}

int8_t piece_moves(Location* moves, const int8_t* board, Location loc) {
    int8_t piece = get_piece(board, loc);
    switch (piece < 0 ? -piece : piece) {
        case PAWN:
            return pawn_moves(moves, board, loc);
        case CANNON:
            return cannon_moves(moves, board, loc);
        case ROOK:
            return rook_moves(moves, board, loc);
        case KNIGHT:
            return knight_moves(moves, board, loc);
        case ELEPHANT:
            return elephant_moves(moves, board, loc);
        case GUARD:
            return guard_moves(moves, board, loc);
        case KING:
            return king_moves(moves, board, loc);
        default:
            return 0;
    }
}

/*

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
    board[3] = -1;
    board[6] = -1;
    //board[4+9*2] = 1;
    board[4+9*9] = -KING;
    Location moves[100];

    board[4] = KING;
    n_moves = piece_moves(moves, board, (Location){0,4});
 
    for(i = 0; i < n_moves; ++i) {
        printf("(%d,%d)\r\n", moves[i].row, moves[i].col);
    }
    return 0;
}