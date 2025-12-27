#ifndef HNSW_CMD_ARGS_H
#define HNSW_CMD_ARGS_H

struct CmdArgs {
    // --- index ---
    int dim = 128;
    int M = 16;
    int efc = 200;

    // --- search ---
    int k = 15;
    int efs = 80;
    int queries = 30;

    // --- clusters / UT ---
    int clusters = 6;
    int pts = 200;
    float sigma = 0.004f;
    float center_dist = 8.0f;
    int seed = 42;

    // --- execution ---
    int threads = 1;   // number of worker threads

    bool ut1 = false;
    bool ut2 = false;
};

void print_usage(const char *prog);
CmdArgs parse_args(int argc, char **argv);

#endif // HNSW_CMD_ARGS_H
