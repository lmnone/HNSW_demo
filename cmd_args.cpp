//
// Created by Konstantin Sobolev on 26/12/2025.
//

#include "cmd_args.h"
#include <iostream>
#include <string>


void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
                                      "Index build:\n"
                                      "  --dim N            vector dimension (128)\n"
                                      "  --M N              HNSW max neighbors (16)\n"
                                      "  --efc N            ef_construction (200)\n\n"
                                      "Search:\n"
                                      "  --k N              KNN K (15)\n"
                                      "  --efs N            ef_search (80)\n"
                                      "  --queries N        queries per cluster (30)\n\n"
                                      "Clusters / UT:\n"
                                      "  --clusters N       number of clusters (6)\n"
                                      "  --pts N            points per cluster (200)\n"
                                      "  --sigma X          intra-cluster sigma (0.004)\n"
                                      "  --center-dist X    center distance (8.0)\n"
                                      "  --seed N           RNG seed (42)\n\n"
                                      "Execution:\n"
                                      "  --threads N        number of threads (1)\n\n"
                                      "Modes:\n"
                                      "  --ut1              HNSW vs exact KNN\n"
                                      "  --ut2              per-cluster precision UT\n"
                                      "\n";
}


static void parse_value(int &v, const char *s) {
    v = std::stoi(s);
}
static void parse_value(float &v, const char *s) {
    v = std::stof(s);
}
static void parse_value(double &v, const char *s) {
    v = std::stod(s);
}

CmdArgs parse_args(int argc, char **argv) {
    CmdArgs a;

    for (int i = 1; i < argc; ++i) {

        auto next = [&](auto &v) {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << argv[i] << "\n";
                std::exit(1);
            }
            parse_value(v, argv[++i]);
        };

        std::string s = argv[i];

        if (s == "--dim")
            next(a.dim);
        else if (s == "--M")
            next(a.M);
        else if (s == "--efc")
            next(a.efc);
        else if (s == "--k")
            next(a.k);
        else if (s == "--efs")
            next(a.efs);
        else if (s == "--queries")
            next(a.queries);
        else if (s == "--clusters")
            next(a.clusters);
        else if (s == "--pts")
            next(a.pts);
        else if (s == "--sigma")
            next(a.sigma);
        else if (s == "--center-dist")
            next(a.center_dist);
        else if (s == "--seed")
            next(a.seed);
        else if (s == "--threads")
            next(a.threads);
        else if (s == "--ut1")
            a.ut1 = true;
        else if (s == "--ut2")
            a.ut2 = true;
        else {
            std::cerr << "Unknown option: " << s << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    // Basic sanity check
    if (a.threads <= 0) {
        std::cerr << "--threads must be > 0\n";
        std::exit(1);
    }

    return a;
}
