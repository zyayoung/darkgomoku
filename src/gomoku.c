#include "gomoku.h"

#ifdef GUI
#include <gtk/gtk.h>
#endif

void tb_record(char *type, float val, int step) {
    static FILE *pf = NULL;
    if (!pf) {
        pf = popen("python logger.py", "w");
        assert(pf);
    }
    fprintf(pf, "%s\n%f\n%d\n", type, val, step);
    fflush(pf);
}

int player_random(Game *game) {
    int legal_pos = 0;
    for (Pos p = 0; p < game->wh; p++) {
        if (!game->board_c[p]) {
            legal_pos += 1;
        }
    }
    Pos rand_pos = 0;

    while (game->board_c[rand_pos]) rand_pos++;
    for (int p = rand_int(0, legal_pos - 1); p; p--) {
        rand_pos++;
        while (game->board_c[rand_pos]) {
            rand_pos++;
        }
    }
    return rand_pos;
}

int player_net_with_prob(float *prob, Game *game) {
    float *prob_cur = prob;
    float legal_prob = 0;
    for (Pos p = 0; p < game->wh; p++) {
        if (!game->board_c[p]) {
            legal_prob += prob_cur[p];
        } else {
            prob_cur[p] = 0;
        }
    }
    Pos rand_pos = 0;
    float p = random_float() * legal_prob - prob_cur[rand_pos];
    while (p > 0 && rand_pos < game->wh) {
        rand_pos++;
        p -= prob_cur[rand_pos];
    }
    if (rand_pos >= game->wh || game->board_c[rand_pos]) {
        return player_random(game);
    }
    if (game->board_c[rand_pos]) return player_random(game);
    return rand_pos;
}

int player_net(network *net, Game *game) {
    float *prob_cur =
        network_predict(*net, game->board[game_next_player(game)]);
    constrain_cpu(game->wh, 12, prob_cur);
    activate_array(prob_cur, game->wh, LOGISTIC);
    print_grid(prob_cur, game->w, game->h);

    float max_val = 0.;
    for (int i = 0; i < game->wh; i++) {
        if (prob_cur[i] > max_val && !game->board_c[i]) {
            max_val = prob_cur[i];
        }
    }
    scal_add_cpu(game->wh, 1. / max_val, 0, prob_cur, 1);

    pow_const_cpu(game->wh, 1. / net->layers[net->n - 1].temperature, prob_cur,
                  1);
    print_grid(prob_cur, game->w, game->h);
    return player_net_with_prob(prob_cur, game);
}

typedef struct val_pos {
    float val;
    int pos;
} val_pos;

val_pos best_move(float *x, Game *g) {
    val_pos ret = {-1, -1};
    for (int i = 0; i < g->wh; i++) {
        if (x[i] > ret.val && !g->board_c[i]) {
            ret = (val_pos){x[i], i};
        }
    }
    return ret;
}

val_pos alpha_beta_helper(network *net, Game *game, int depth, int alpha,
                          int beta) {
    float *prob = network_predict(*net, game->board);
    float prob_cur[game->wh];
    memcpy(prob_cur, prob + !(game->cur_node->player) * game->wh,
           game->wh * sizeof(float));
    activate_array(prob_cur, game->wh, LOGISTIC);
    if (depth == 0) {
        return best_move(prob_cur, game);
    }
    int pos = -1;
    int prob_thr = -1;
    for (int i = 0; i < 2; i++) {
        int move = best_move(prob_cur, game).pos;
        // if(prob_thr == -1) prob_thr = prob_cur[move] * 0.9;
        if (prob_cur[move] < prob_thr || move < 0) break;
        prob_cur[move] = -1;
        game_place(game, move);
        if (game_end(game) != 3) {
            game_unplace(game);
            return (val_pos){1, move};
        }
        float val = -alpha_beta_helper(net, game, depth - 1, -beta, -alpha).val;
        game_unplace(game);
        if (val >= beta) {
            return (val_pos){beta, move};
        }
        if (val > alpha) {
            alpha = val;
            pos = move;
        }
    }
    return (val_pos){alpha, pos};
}

int player_net_alpha_beta(network *net, Game *game, int depth) {
    return alpha_beta_helper(net, game, depth, -10, 10).pos;
}

// self play gomoku (for testing only)
float self_gomoku_loop(network *net, network *net_ref) {
    int num_win = 0;
    int num_lose = 0;

    // allocate memory
    static Game *game;
    static Game *game_tmp = NULL;
    static float *board_tmp = NULL;
    if (game_tmp == NULL) {
        game = (Game *)xcalloc(1, sizeof(Game));
        game_init(game, net);
        game_tmp = (Game *)xcalloc(net->round, sizeof(Game));
        board_tmp = (float *)xcalloc(3 * game->wh * net->round, sizeof(float));
        for (int r = 0; r < net->round; r++) game_tmp_init(game_tmp + r, net);
    }

    // prepare batch
    for (int iter = 0; iter < 16; iter++) {
        int sbatch_map[net->round];
        int net_play = iter % 2;
        for (int hand = 0; 1; hand++) {
            int sbatch_size = 0;
            for (int r = 0; r < net->round; r++) {
                if (game_tmp[r].winner == 3) {
                    memcpy(board_tmp + sbatch_size * 3 * game->wh,
                           game_tmp[r].board[game_next_player(game_tmp + r)],
                           3 * game->wh * sizeof(float));
                    sbatch_map[sbatch_size++] = r;
                }
            }
            if (!sbatch_size) break;
            network *net_this = (net_play ^ (hand % 2)) ? net_ref : net;
            set_batch_network(net_this, sbatch_size);
            float *prob = network_predict(*net_this, board_tmp);
            for (int s = 0; s < sbatch_size; s++) {
                int r = sbatch_map[s];
                float *prob_cur = prob + s * game->wh;
                constrain_cpu(game->wh, 12, prob_cur);
                activate_array(prob_cur, game->wh, LOGISTIC);
                float max_val = 0;
                for (int i = 0; i < game->wh; i++) {
                    if (prob_cur[i] > max_val && !game->board_c[i]) {
                        max_val = prob_cur[i];
                    }
                }
                scal_add_cpu(game->wh, 1. / max_val, 0, prob_cur, 1);
                pow_const_cpu(game->wh, 200, prob_cur, 1);
                game_place(game_tmp + r,
                           player_net_with_prob(prob_cur, game_tmp + r));
                game_tmp[r].winner = game_end(game_tmp + r);
            }
        }
        for (int r = 0; r < net->round; r++)
            num_win += game_tmp[r].winner == net_play;
        for (int r = 0; r < net->round; r++)
            num_lose += game_tmp[r].winner == !net_play;
        for (int r = 0; r < net->round; r++) game_reset(game_tmp + r);
    }

    printf("Win: %d  lose: %d\n", num_win, num_lose);
    return num_win * 100.0 / (num_win + num_lose);
}

#ifdef GPU
void train_single_iter(network *net) {
    float loss_sum = 0, conf_sum = 0;
    int ndraw = 0;
    int first_win = 0;
    static float first_win_rate = 0.5;

    // allocate memory
    Game game_tmp[net->round], _game;
    Game *game = &_game;
    game_init(game, net);
    float board_tmp[3 * game->wh * net->round];
    for (int r = 0; r < net->round; r++) game_tmp_init(game_tmp + r, net);

    // prepare batch
    game->memory_pos = 0;
    int sbatch_map[net->round];
    for (int hand = 0; 1; hand++) {
        int sbatch_size = 0;
        for (int r = 0; r < net->round; r++) {
            if (game_tmp[r].winner == 3) {
                memcpy(board_tmp + sbatch_size * 3 * game->wh,
                       game_tmp[r].board[game_next_player(game_tmp + r)],
                       3 * game->wh * sizeof(float));
                sbatch_map[sbatch_size++] = r;
            }
        }
        if (!sbatch_size) break;

        // self play step
        set_batch_network(net, sbatch_size);
        float *prob = network_predict(*net, board_tmp);
        for (int s = 0; s < sbatch_size; s++) {
            int r = sbatch_map[s];
            float *prob_cur = prob + s * game->wh;
            constrain_cpu(game->wh, 12, prob_cur);
            activate_array(prob_cur, game->wh, LOGISTIC);
            float max_val = 0;
            for (int i = 0; i < game->wh; i++) {
                if (prob_cur[i] > max_val && !game->board_c[i]) {
                    max_val = prob_cur[i];
                }
            }
            scal_add_cpu(game->wh, 1. / max_val, 0, prob_cur, 1);
            float pow = 1. / net->layers[net->n - 1].temperature;
            pow = fmin(pow, *net->cur_iteration * 5e-4);
            pow_const_cpu(game->wh, pow, prob_cur, 1);
            game_place(game_tmp + r,
                       player_net_with_prob(prob_cur, game_tmp + r));
            game_tmp[r].winner = game_end(game_tmp + r);
        }
    }

    // perpare training data
    for (int r = 0; r < net->round; r++) {
        Game *this_game = game_tmp + r;
        if (this_game->winner == 2) {
            ndraw++;
        }
        first_win += this_game->winner == 0;
        for (int rev_i = 0; this_game->cur_node != this_game->root; rev_i++) {
            game->player_memory[game->memory_pos] = this_game->cur_node->player;
            game->winner_memory[game->memory_pos] = this_game->winner;

            Pos pos = game_unplace(this_game);
            float *mem = game->board_memory + game->wh * 3 * game->memory_pos;
            memcpy(mem, this_game->board[!this_game->cur_node->player],
                   3 * game->wh * sizeof(float));

            // data argumentation (rotation and flip)
            int x = pos % game->w, y = pos / game->w;
            int flip = rand() % 2;
            int rotate = rand() % 4;
            image board_im = float_to_image(game->w, game->h, 2, mem);
            if (flip) {
                flip_image(board_im);
                x = game->w - 1 - x;
            }
            rotate_image_cw(board_im, rotate);
            for (; rotate; rotate--) {
                int swap = x;
                x = y;
                y = game->w - 1 - swap;
            }
            pos = x + y * game->w;

            game->pos_memory[game->memory_pos] = pos;
            game->iter_memory[game->memory_pos] = game->n_playout;
            game->memory_pos++;
        }
        game->n_playout++;
    }
    for (int r = 0; r < net->round; r++) free_game(game_tmp + r);

    float p_target[2] = {0};
    int cur_iter = -1;
    layer *p_layer = &net->layers[net->n - 1];
    network_state state = {0};
    state.net = *net;
    state.train = 1;

    for (int s = 0; s < game->memory_pos; s += net->max_batchsize) {
        int batch_size = game->memory_pos - s;
        if (batch_size > net->max_batchsize) batch_size = net->max_batchsize;
        set_batch_network(net, batch_size);

        state.input = cuda_make_array(game->board_memory + s * 3 * game->wh,
                                      batch_size * 3 * game->wh);
        forward_network_gpu(*net, state);

        for (int i = 0; i < batch_size; i++) {
            uint8_t winner = game->winner_memory[i + s];
            uint8_t player = game->player_memory[i + s];
            const Pos pos = game->pos_memory[i + s];
            if (game->iter_memory[i + s] != cur_iter) {
                if (winner == 2) {
                    p_target[0] = 0;
                    p_target[1] = 0;
                } else {
                    p_target[winner] = 1;
                    p_target[!winner] = 0;
                }
                cur_iter = game->iter_memory[i + s];
            }

            float p_delta, p_pred;

            // p_pred
            cuda_pull_array(p_layer->output_gpu + i * game->wh + pos, &p_pred,
                            1);
            p_pred = 1 / (1 + expf(-constrain(-12, 12, p_pred)));

            float c_entropy = -log(p_pred) * p_target[player] -
                              log(1 - p_pred) * (1 - p_target[player]);
            loss_sum += c_entropy;
            p_delta = (p_target[player] - p_pred);
            cuda_push_array(p_layer->delta_gpu + i * game->wh + pos, &p_delta,
                            1);

            p_target[player] = p_pred;
        }
        backward_network_gpu(*net, state);
        cuda_free(state.input);
    }
    (*net->seen) += game->memory_pos;
    
    int batch_latent = net->batch;
    net->batch = game->memory_pos;
    update_network_gpu(*net);
    net->batch = batch_latent;

    if (net->record_next) {
        first_win_rate =
            0.9 * first_win_rate + 0.1 * (float)first_win / net->round;
        tb_record("first_win_rate", first_win_rate, *net->cur_iteration);
        tb_record("ndraw", (float)ndraw / net->round, *net->cur_iteration);
        tb_record("avg_step", (float)game->memory_pos / net->round,
                  *net->cur_iteration);
        tb_record("loss", loss_sum / game->memory_pos, *net->cur_iteration);
        printf("%8d, %f loss, %f lr\n", *net->cur_iteration, loss_sum,
               net->learning_rate);
        net->record_next = 0;
    }
    free_game(game);
}

void train_gomoku(char *cfgfile, char *weightfile, int *gpus, int ngpus) {
    srand(time(0));
    int seed = rand();

    float avg_time = -1;
    char *base = basecfg(cfgfile);
    printf("%s\n", base);

    network *nets = (network *)xcalloc(ngpus, sizeof(network));
    network *net_infs = (network *)xcalloc(ngpus, sizeof(network));
    pthread_t threads[ngpus];

    for (int i = 0; i < ngpus; ++i) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[i]);
#endif
        nets[i] = parse_network_cfg(cfgfile);
        nets[i].index = i;
        if (weightfile) {
            set_batch_network(&nets[i], 1);
            load_weights(&nets[i], weightfile);
        }
        nets[i].learning_rate *= ngpus;
    }
    // *nets->cur_iteration = 0;
    cuda_set_device(gpus[0]);
    network net_ref =
        parse_network_cfg_custom("cfg/gomoku_ref.cfg", nets->round, 0);
    load_weights(&net_ref, "backup/gomoku_ref.w");
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", nets->learning_rate,
           nets->momentum, nets->decay);

    while (*nets->cur_iteration <= nets->max_batches) {
        double time = what_time_is_it_now();
        // train_single_iter(&net);
        for (int i = 0; i < ngpus; ++i)
            if (pthread_create(threads + i, 0, train_single_iter, nets + i))
                error("Thread creation failed");
        for (int i = 0; i < ngpus; ++i) pthread_join(threads[i], 0);

        if (get_current_iteration(nets[0]) % (4 * ngpus) == 0) {
            printf("Syncing... ");
            fflush(stdout);
            sync_nets(nets, ngpus, 4);
            for (int j = 1; j < ngpus; j++)
                *nets[j].cur_iteration = *nets->cur_iteration;
            printf("Done!\n");
        }
        if (avg_time == -1) avg_time = what_time_is_it_now() - time;
        float time_remaining = (nets->max_batches - *nets->cur_iteration) *
                               avg_time / 60 / 60 / ngpus;
        printf("eta: %fh\n", time_remaining);

        (*nets->cur_iteration) += ngpus;
        nets->record_next = *nets->cur_iteration % (10 * ngpus) == 0;
        if (*nets->cur_iteration % 10000 == 0) {
            char buff[256];
            sprintf(buff, "backup/%s_%08d.backup", base, *nets->cur_iteration);
            set_batch_network(nets, 1);
            save_weights(*nets, buff);
        }
        if (*nets->cur_iteration % 1000 == 0) {
            float num_win = self_gomoku_loop(nets, &net_ref);
            tb_record("win", num_win, *nets->cur_iteration);
        }
        avg_time = avg_time * .99 + (what_time_is_it_now() - time) * .01;
    }
    for (int i = 0; i < ngpus; ++i) free_network(nets[i]);
    free(nets);
    free(net_infs);
    free_network(net_ref);
    free(base);
}
#else
void train_gomoku(char *cfgfile, char *weightfile, int *gpus, int ngpus) {}
#endif

#ifdef GUI
typedef struct game_info game_info;
struct game_info {
    network *net;
    Game *game;
    Pos pos;
    game_info *root;
    GtkWidget **window;
    GtkWidget **vbox;
    GtkWidget **grid;
};

static void on_click(GtkWidget *widget, game_info *info) {
    GtkWidget *button;
    if (!*info->window) {
        *info->window = gtk_application_window_new(widget);
        *info->vbox = gtk_vbox_new(FALSE, 8);

        GtkWidget *hbox = gtk_hbox_new(FALSE, 8);

        button = gtk_button_new_with_label("New Game");
        g_signal_connect(button, "clicked", G_CALLBACK(on_click),
                         info->root + info->game->wh);
        gtk_container_add(GTK_CONTAINER(hbox), button);

        button = gtk_button_new_with_label("Auto Step");
        g_signal_connect(button, "clicked", G_CALLBACK(on_click),
                         info->root + info->game->wh + 1);
        gtk_container_add(GTK_CONTAINER(hbox), button);

        gtk_container_add(GTK_CONTAINER(*info->vbox), hbox);
        gtk_container_add(GTK_CONTAINER(*info->window), *info->vbox);
    } else {
        gtk_container_remove(GTK_CONTAINER(*info->vbox), *info->grid);
    }
    *info->grid = gtk_grid_new();

    if (info->pos < info->game->wh) {
        game_place(info->game, info->pos);
        if (game_end(info->game) == 3)
            game_place(info->game, player_net(info->net, info->game));
    } else if (info->pos == info->game->wh) {
        game_reset(info->game);
    } else if (info->pos == info->game->wh + 1) {
        if (game_end(info->game) == 3)
            game_place(info->game, player_net(info->net, info->game));
    }

    float *prob_cur = network_predict(
        *info->net, info->game->board[game_next_player(info->game)]);
    activate_array(prob_cur, info->game->wh, LOGISTIC);

    for (int i = 0; i < info->game->wh; i++) {
        if (info->game->board_c[i]) {
            button = gtk_button_new_with_label(
                info->game->board_c[i] == 1 ? "x" : "o");
        } else {
            char tmp[10];
            sprintf(tmp, "%.2f", prob_cur[i]);
            button = gtk_button_new_with_label(tmp);
            gtk_window_set_opacity(button, 0.25);
            // gtk_color_button_set_alpha(button, 127);
            if (game_end(info->game) == 3)
                g_signal_connect(button, "clicked", G_CALLBACK(on_click),
                                 info->root + i);
        }
        gtk_grid_attach(GTK_GRID(*info->grid), button, i % info->game->w,
                        i / info->game->w, 1, 1);
    }
    gtk_container_add(GTK_CONTAINER(*info->vbox), *info->grid);
    gtk_widget_show_all(*info->window);
}

// play gomoku with trained model on gui
void play_gomoku(char *cfgfile, char *weightfile) {
    // Create a new application
    GtkApplication *app = gtk_application_new("com.example.GtkApplication",
                                              G_APPLICATION_FLAGS_NONE);

    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    assert(weightfile);
    load_weights(&net, weightfile);

    Game game;
    game_init(&game, &net);

    game_info infos[game.wh + 2];
    GtkWidget *window = NULL;
    GtkWidget *grid;
    GtkWidget *vbox;
    for (int i = 0; i < game.wh + 2; i++)
        infos[i] = (game_info){&net, &game, i, infos, &window, &vbox, &grid};

    g_signal_connect(app, "activate", G_CALLBACK(on_click), infos + game.wh);
    g_application_run(G_APPLICATION(app), 0, NULL);
}

#else

// watch step by step self play
void play_gomoku(char *cfgfile, char *weightfile) {
    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    network net = parse_network_cfg_custom(cfgfile, 1, 0);
    assert(weightfile);
    load_weights(&net, weightfile);

    Game game;
    game_init(&game, &net);

    int net_play = 1;
    // net.layers[net.n-1].temperature = 0.1;
    while (game_end(&game) == 3) {
        // if ((!game.cur_node->player) == net_play){
        //     game_place(&game, player_net(&net, &game));
        // } else {
        //     int x, y;
        //     print_grid_char(game.board_c, game.w, game.h);
        //     do
        //     {
        //         printf("Choose a pos:\n");
        //         scanf("%d %d", &y, &x);
        //     } while (game.board_c[x + y * game.w]);
        //     game_place(&game, x + y * game.w);
        // }
        game_place(&game, player_net(&net, &game));
        print_grid_char(game.board_c, game.w, game.h);
        getchar();
    }
    int winner = game_end(&game);

    printf("\nWinenr is : %d\n", winner);

    free_network(net);
    free_game(&game);
    free(base);
}

#endif

void run_gomoku(int argc, char **argv) {
    // boards_gomoku();
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s %s [train/play] [cfg] [weights (optional)]\n",
                argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];

    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int *)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    char *weights = (argc > 4) ? argv[4] : 0;
    char *weights_ref = (argc > 5) ? argv[5] : 0;
    if (0 == strcmp(argv[2], "train"))
        train_gomoku(cfg, weights, gpus, ngpus);
    else if (0 == strcmp(argv[2], "play"))
        play_gomoku(cfg, weights);
}
