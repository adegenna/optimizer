#pragma once

struct PartSwarmParams {
    int n_swarm_;
    std::function<bool( const std::vector<std::vector<float>>&)> is_done_;
    std::function<std::vector<float>()> draw_random_state_;
    std::function<std::vector<float>()> draw_random_vel_;
    float w_;
    float phi_p_;
    float phi_g_;

};