#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "gomoku.h"
#include "image.h"

float print_grid(float *b, int w, int h){
    for (int j=0; j<w; j++) printf("---");
    printf("\n");
    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            printf("%f ", *(b+j+i*w));
        }
        printf("\n");
    }
}

float print_grid_char(uint8_t *b, int w, int h){
    for (int j=0; j<w; j++) printf("--");
    printf("-\n");
    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            if(*(b+j+i*w)==0) printf("  ");
            else if(*(b+j+i*w)==1) printf(" x");
            else if(*(b+j+i*w)==2) printf(" o");
            else printf(" ?");
        }
        printf("\n");
    }
    for (int j=0; j<w; j++) printf("--");
    printf("-\n");
}

MCTreeNode *node_create(){
    MCTreeNode *n = (MCTreeNode*)xcalloc(1, sizeof(MCTreeNode));
    return n;
}

MCTreeNode *node_find_child_or_create(MCTreeNode *n, Pos p){
    if(n->children == NULL){
        n->children = node_create();
        n->children->parent = n;
        n->children->depth = n->depth + 1;
        n->children->player = !n->player;
    }
    n->children->pos = p;
    return n->children;
}


// Effect: return i if player i win, 2 if equal and 3 if not ended
int game_end(Game *g){
    if(g->cur_node->depth == g->wh) return 2;
    Pos pos = g->cur_node->pos;
    int dx[4] = {0, 1, 1, 1};
    int dy[4] = {1, 0, 1, -1};
    for (int d=0; d<4; d++){
        int cnt=0;
        uint8_t x = pos % g->w, y = pos / g->w;
        for(; 0<=x && x<g->w && 0<=y && y<g->h; x+=dx[d], y+=dy[d]){
            if(g->board_c[x+y*g->w] == g->cur_node->player+1)cnt++;
            else break;
        }
        x = pos % g->w - dx[d], y = pos / g->w - dy[d];
        for(; 0<=x && x<g->w && 0<=y && y<g->h; x-=dx[d], y-=dy[d]){
            if(g->board_c[x+y*g->w] == g->cur_node->player+1)cnt++;
            else break;
        }
        if (cnt >= g->nwin) return g->cur_node->player;
    }
    return 3;
}

void game_reset(Game *g){
    g->cur_node = g->root;
    memset(g->board_c, 0, g->wh * sizeof(uint8_t));
    memset(g->board[0], 0, g->wh * 2 * sizeof(float));
    memset(g->board[1], 0, g->wh * 2 * sizeof(float));
    g->winner = 3;
}

void game_place(Game *g, Pos p){
    g->cur_node = node_find_child_or_create(g->cur_node, p);
    g->board[0][p + g->wh * g->cur_node->player] = 1;
    g->board[1][p + g->wh * !g->cur_node->player] = 1;
    assert(!g->board_c[p]);
    g->board_c[p] = g->cur_node->player + 1;
}

Pos game_unplace(Game *g){
    Pos p = g->cur_node->pos;
    g->board[0][p + g->wh * g->cur_node->player] = 0;
    g->board[1][p + g->wh * !g->cur_node->player] = 0;
    g->board_c[p] = 0;
    g->cur_node = g->cur_node->parent;
    return p;
}

void free_game(Game *g){
    free(g->board_c);
    free(g->board[0]);
    free(g->board[1]);
    free(g->board_memory);
    free(g->player_memory);
    free(g->winner_memory);
    free(g->pos_memory);
    free(g->iter_memory);
    for(MCTreeNode *n=g->root; n; n=n->children){
        free(n);
    }
}

void game_init_with_memory(Game *g, int w, int h, int nwin, int memory_size){
    g->w = w;
    g->h = h;
    g->wh = w * h;
    g->nwin = nwin;
    g->board_c = (uint8_t*)xcalloc(w * h, sizeof(uint8_t));
    g->board[0] = (float*)xcalloc(w * h * 3, sizeof(float));
    g->board[1] = (float*)xcalloc(w * h * 3, sizeof(float));
    for(int i=0; i<g->w; i++){
        g->board[0][2*g->wh+i] = 1;
        g->board[0][2*g->wh+i*g->w] = 1;
        g->board[0][2*g->wh+i*g->w+g->w-1] = 1;
        g->board[0][2*g->wh+(g->h-1)*g->w + i] = 1;
        g->board[1][2*g->wh+i] = 1;
        g->board[1][2*g->wh+i*g->w] = 1;
        g->board[1][2*g->wh+i*g->w+g->w-1] = 1;
        g->board[1][2*g->wh+(g->h-1)*g->w + i] = 1;
    }
    g->memory_size = memory_size;
    g->n_playout = 0;
    g->board_memory = (float*)xcalloc(3 * w * h * g->memory_size, sizeof(float));
    g->player_memory = (uint8_t*)xcalloc(g->memory_size, sizeof(uint8_t));
    g->winner_memory = (uint8_t*)xcalloc(g->memory_size, sizeof(uint8_t));
    g->pos_memory  = (int*)xcalloc(g->memory_size, sizeof(int));
    g->iter_memory  = (int*)xcalloc(g->memory_size, sizeof(int));
    g->init_done = 0;
    g->memory_pos = 0;
    g->root = node_create();
    g->root->player = 1;
    g->root->parent = g->root;
    g->cur_node = g->root;
    g->winner = 3;
}

void game_init(Game *g, network *net){
    game_init_with_memory(g, net->w, net->h, net->nwin, net->round * net->w * net->h);
}

void game_tmp_init(Game *g, network *net){
    game_init_with_memory(g, net->w, net->h, net->nwin, 0);
}

int game_next_player(Game *g) { return !g->cur_node->player; }
