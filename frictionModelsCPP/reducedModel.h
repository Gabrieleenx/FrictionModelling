#pragma once

#include <vector>
#include "utils.h"
#include "distributedModel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


class PreCompute{
    private:
        int nr_segments;
        double ra;
        double tau_visc;
        std::vector<double> cop;
        std::vector<double> pos;
        std::vector<utils::four_points> ls_list;
        DistributedFrictionModel distributed_model;
        DistributedFrictionModel distributed_model_viscus;
        utils::closest_sample get_closest_samples(utils::vec vel, double ra_); 
        void pre_compute_pos();
        utils::if_calc get_if_calc(int r, int i, int j);
        utils::vec calc_vel(int r, int i, int j);
        int calc_idx(int r, int i);
        void calc_4_points(int r, int i, double f_t_max, double f_tau_max);
        std::vector<double> normalize_force(std::vector<double> f, double f_t_max, double f_tau_max);
        void update_viscus_scale();

    public:
        PreCompute(){};
        void update_surface(std::vector<double> new_surf, std::string shape_name, double fn);
        std::vector<double> get_bilinear_interpolation(utils::vec vel, double ra_);
        
        std::vector<double> calc_skew_var(utils::vec vel_at_cop, double ra_);
        
        void update_distributed_model(pybind11::list py_list, std::string shape_name);

        void pre_comp_ls(int nr_segments_);

        double calc_ra();

        std::vector<double> get_viscus_scale(double ra_);

};


class ReducedFrictionModel{
    private:
        double ra;
        double N_LS;
        std::vector<double> viscus_scale;
        std::vector<double> force_vec_cop;
        std::vector<double> force_at_center;

        utils::vec velocity;
        utils::P_x_y p_x_y;
        utils::shape_info_red shape_info_var_red;
        utils::properties properties; 
        utils::lugre_red lugre;

        PreCompute pre_compute;

        void update_lugre(utils::vec vel_cop);

        std::vector<double> move_force_to_center(std::vector<double> force_at_cop);

        void calc_w(utils::vec vel_cop, double& v_n, std::vector<double>& Av, std::vector<double>& w);
    public:
        ReducedFrictionModel(){};
        void init(pybind11::list py_list, std::string shape_name, double fn);
        std::vector<double> step(pybind11::list py_list);
        std::vector<double> step_ode(pybind11::list py_y, pybind11::list py_vel);
        std::vector<double> get_force_at_cop();
        std::vector<double> get_force_at_center();
        std::vector<double> get_cop();
        void set_fn(double fn){p_x_y.set_fn(fn);};
        void update_surface(pybind11::list py_list, std::string shape_name, double fn, int recalc);
};

