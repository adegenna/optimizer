#pragma once

struct NewtRhapParams {
    std::vector<float> x0_;
    int maxIters_;
};

struct GradDesParams {
    std::vector<float> x0_;
    int maxIters_;
    float scale_;
};

struct GradDesMomParams {
    std::vector<float> x0_;
    int maxIters_;
    float scale_;
    float momentum_;
};

struct SimAnnParams {
    int maxIters_;
    std::vector<float> x0_;
    std::function<float(float)> f_annealing_;
    std::function<std::vector<float>(const std::vector<float>&)> f_random_neighbor_;
    std::function<float(float,float,float)> prob_accept_;
};

struct PartSwarmParams {
    int maxIters_;
    int n_swarm_;
    std::function<bool( const std::vector<std::vector<float>>&)> is_done_;
    std::function<std::vector<float>()> draw_random_state_;
    std::function<std::vector<float>()> draw_random_vel_;
    float w_;
    float phi_p_;
    float phi_g_;
};

