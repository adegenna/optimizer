#pragma once

#include <string>
#include <memory>

#include "Function.hpp"
#include "Data.hpp"
#include "Parameters.hpp"

class Optimizer {

    public:

        virtual void solve() = 0;
        void output_solver_history( const std::string outFileName ) { solver_data_->output_history(outFileName); }
        float get_last_cost() { return solver_data_->get_last_cost(); }
        std::vector<float> get_last_state() { return solver_data_->get_last_state(); }

    protected:

        Optimizer( const Function& cost , std::unique_ptr<DataOptimizer> dptr ) : cost_(cost) , solver_data_(std::move(dptr)) {};
        const Function& cost_;
        std::unique_ptr<DataOptimizer> solver_data_;

};


class NewtonRhapson : public Optimizer {

    public:
        
        NewtonRhapson( const Function& cost , NewtRhapParams p ) : Optimizer( cost , std::make_unique<DataOptimizer_singleInitialValue>() ) , p_(p) {};
        void solve();

    private:
        
        NewtRhapParams p_;

};

class GradientDescent : public Optimizer {

    public:

        GradientDescent( const Function& cost , GradDesParams p ) : Optimizer( cost , std::make_unique<DataOptimizer_singleInitialValue>() ) , p_(p) {};
        void solve();

    private:

        GradDesParams p_;

};


class GradientDescentMomentum : public Optimizer {

    public:

        GradientDescentMomentum( const Function& cost , GradDesMomParams p ) : Optimizer( cost , std::make_unique<DataOptimizer_singleInitialValue>() ) , p_(p) {};
        void solve();

    private:

        GradDesMomParams p_;

};


class SimulatedAnnealing : public Optimizer {

    public:

        SimulatedAnnealing( const Function& cost , SimAnnParams s ) : Optimizer( cost , std::make_unique<DataOptimizer_singleInitialValue>() ) , sim_ann_params_(s) {};
        void solve();

    private:

        SimAnnParams sim_ann_params_;

};


class ParticleSwarm : public Optimizer {

    public:

        ParticleSwarm( const Function& cost , PartSwarmParams s ) : Optimizer( cost , std::make_unique<DataOptimizer_ParticleMethod>() ) , params_(s) { set_initial_data_(); };
        void solve();

    private:

        PartSwarmParams params_;
        std::vector<std::vector<float>> x_;
        std::vector<std::vector<float>> vel_;
        std::vector<std::vector<float>> p_;
        std::vector<float> g_;
        std::vector<float> J_p_;
        float J_g_;
        void set_initial_data_();

};