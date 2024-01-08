#pragma once

#include <vector>
#include "utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

class DistributedFrictionModel{
    private:
        utils::vec velocity;
        utils::P_x_y p_x_y;
        utils::shape_info shape_info_var;
        utils::properties properties; 
        utils::lugre lugre;
        std::vector<std::vector<std::vector<double>>> vel_grid; // [shape x, shape y, 2]
        std::vector<double> position_vec_x; // [shape x]
        std::vector<double> position_vec_y; // [shape x]

        std::vector<double> step_single_point();
        std::vector<double> step_bilinear();
        std::vector<double> force_vec_cop;
        std::vector<double> force_vec_center;


        void update_velocity_grid(utils::vec vel);
        void update_lugre();
        std::vector<double> approximate_integral();
        std::vector<double> move_force_to_cop(std::vector<double> force_at_center);

    public:
        DistributedFrictionModel(){};
        void init(pybind11::list py_list, std::string shape_name, double fn);
        void init_cpp(utils::properties properties_, std::string shape_name, double fn);
        std::vector<double> step(pybind11::list py_list);
        std::vector<double> step_ode(pybind11::list py_y, pybind11::list py_vel);
        std::vector<double> step_cpp(utils::vec vel);
        std::vector<double> get_cop();
        utils::properties get_properties();
        std::vector<double> get_force_at_cop();
        std::vector<double> get_force_at_center();
        void update_surface(pybind11::list py_list, std::string shape_name, double fn);
        void update_surface_cpp(std::vector<double> new_surf, std::string shape_name, double fn);
        void set_fn(double fn){p_x_y.set_fn(fn);};
};

