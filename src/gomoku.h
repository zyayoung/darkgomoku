#ifndef GOMOKU_H
#define GOMOKU_H

#include <stdlib.h>

#include "blas.h"
#include "image.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "utils.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MCTreeNode MCTreeNode;
typedef struct Game Game;
typedef int Pos;

struct MCTreeNode {
    MCTreeNode *parent;
    MCTreeNode *children;
    int depth;
    uint8_t player;
    Pos pos;
};

struct Game {
    MCTreeNode *root;
    MCTreeNode *cur_node;
    float c_puct;
    int n_playout, w, h, wh, nwin;
    uint8_t *board_c;
    uint8_t winner;
    float *board[2];
    float *board_memory;
    uint8_t *player_memory;
    uint8_t *winner_memory;
    int *iter_memory;
    int *pos_memory;
    int memory_pos, memory_size;
    int init_done;
};

float print_grid(float *b, int w, int h);
float print_grid_char(uint8_t *b, int w, int h);
MCTreeNode *node_create();
MCTreeNode *node_find_child_or_create(MCTreeNode *n, Pos p);

void game_init_with_memory(Game *g, int w, int h, int nwin, int memory_size);
void game_init(Game *g, network *net);
int game_next_player(Game *g);
void free_game(Game *g);
void game_tmp_init(Game *g, network *net);
int game_end(Game *g);
void game_reset(Game *g);
void game_place(Game *g, Pos p);
Pos game_unplace(Game *g);

#ifdef __cplusplus
}
#endif
#endif
