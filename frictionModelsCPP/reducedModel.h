#pragma once

#include <vector>
#include "utils.h"
#include "fullModel.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>




class PreCompute{
    private:
        int nr_segments;
        double gamma;
        std::vector<double> cop;
        std::vector<utils::four_points> ls_list;
        FullFrictionModel full_model;
        utils::closest_sample get_closest_samples(utils::vec vel, double gamma_); 
        void pre_compute_pos();
        utils::if_calc get_if_calc(int r, int i, int j);
        utils::vec calc_vel(int r, int i, int j);
        int calc_idx(int r, int i);
        void calc_4_points(int r, int i, double f_t_max, double f_tau_max);
        std::vector<double> normalize_force(std::vector<double> f, double f_t_max, double f_tau_max);

    public:
        PreCompute(){};

        std::vector<double> get_bilinear_interpolation(utils::vec vel, double gamma_);
        
        utils::vec calc_new_vel(utils::vec vel_at_cop, double gamma_);

        void update_full_model(pybind11::list py_list, std::string shape_name);

        void pre_comp_ls(int nr_segments_);

        double calc_gamma();

        std::vector<double> update_viscus_scale();


};


class ReducedFrictionModel{
    private:
        double gamma;
        std::vector<double> viscus_scale;
        
        utils::vec velocity;
        utils::P_x_y p_x_y;
        utils::shape_info shape_info_var;
        utils::properties properties; 
        utils::lugre_red lugre;

        PreCompute pre_compute;
        
        



        //std::vector<double> position_vec_x; // [shape x]
        //std::vector<double> position_vec_y; // [shape x]

        //void update_velocity_grid(utils::vec vel);
        //void update_lugre();
        //std::vector<double> approximate_integral();
        //std::vector<double> move_force_to_cop(std::vector<double> force_at_center);

    public:
        ReducedFrictionModel(){};
        void init(pybind11::list py_list, std::string shape_name, double fn);
        std::vector<double> step(pybind11::list py_list);


};

