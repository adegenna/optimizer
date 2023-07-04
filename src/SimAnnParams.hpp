#pragma once

struct SimAnnParams {
    std::function<float(float)> f_annealing_;
    std::function<std::vector<float>(const std::vector<float>&)> f_random_neighbor_;
    std::function<float(float,float,float)> prob_accept_;
};