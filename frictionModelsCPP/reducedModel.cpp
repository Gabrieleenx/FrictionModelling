# include "reducedModel.h"
#include <iostream>
#include <algorithm>
#include <cmath>

utils::closest_sample PreCompute::get_closest_samples(utils::vec vel, double gamma_){

    // Temp
    utils::closest_sample sample;
    std::vector<double> f = {0.0, 1.0, 0.1};
    sample.force.f1 = f;
    sample.force.f2 = f;
    sample.force.f3 = f;
    sample.force.f4 = f;
    sample.dr = 0.0;
    sample.di = 1.0;
    return sample;
} 

void PreCompute::pre_compute_pos(){

}

utils::if_calc PreCompute::get_if_calc(int r, int i, int j){
    utils::if_calc calc;
    calc.state = false;
    int idx;
    if (j==0){
        if (i > 0){
            idx = calc_idx(r, i-1);
            calc.value = ls_list[idx].f3;
            calc.state = true;
            return calc; 
        } else if (r > 0){
            idx = calc_idx(r-1, i);
            calc.value = ls_list[idx].f2;
            calc.state = true;
            return calc; 
        }
    }else if (j == 1){
        if (i > 0){
            idx = calc_idx(r, i-1);
            calc.value = ls_list[idx].f4;
            calc.state = true;
            return calc; 
        }
    }else if (j == 2){
        if (r > 0){
            idx = calc_idx(r-1, i);
            calc.value = ls_list[idx].f4;
            calc.state = true;
            return calc; 
        }
    }

    return calc;
}

utils::vec PreCompute::calc_vel(int r, int i, int j){
    double r1;
    double d;
    double v;
    utils::vec vel;

    if (j==0 || j==1){
        r1 = std::tan(i * M_PI / (2 * nr_segments));
    }else{
        r1 = std::tan((i+1) * M_PI / (2 * nr_segments));
    }

    if (j==0 || j==2){
        d = 2*M_PI*r/(4*nr_segments);
    }else{
        d = 2*M_PI*(r+1)/(4*nr_segments);
    }

    vel.tau = std::sqrt(1/(1+pow(r1, 2)));
    v = gamma * std::sqrt(1 - pow(vel.tau, 2));
    vel.x = std::cos(d) * v;
    vel.y = std::sin(d) * v;
    return vel;
}

int PreCompute::calc_idx(int r, int i){
    return r*nr_segments + i;
}


void PreCompute::calc_4_points(int r, int i, double f_t_max, double f_tau_max){
    int idx = 0;
    utils::four_points f;
    utils::if_calc if_calc;
    utils::vec vel;

    idx = calc_idx(r, i);
    for (int j = 0; j <4; j++){
        std::vector<double> fj;
        if_calc = get_if_calc(r, i, j);
        if (if_calc.state == true){
            fj = if_calc.value;
        } else{
            vel = calc_vel(r, i, j);
            full_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), vel));
            fj = normalize_force(full_model.get_force_at_cop(), f_t_max, f_tau_max);    
        }
        if (j==0){f.f1 = fj;}
        if (j==1){f.f2 = fj;}
        if (j==2){f.f3 = fj;}
        if (j==3){f.f4 = fj;}
    }

    ls_list[idx] = f;
}

std::vector<double> PreCompute::normalize_force(std::vector<double> f, double f_t_max, double f_tau_max){
    std::vector<double> fnorm(3);
    fnorm[0] = f[0]/f_t_max;
    fnorm[1] = f[1]/f_t_max;
    fnorm[2] = f[2]/f_tau_max;
    return fnorm;
}

std::vector<double> PreCompute::get_bilinear_interpolation(utils::vec vel, double gamma_){
    std::vector<double> ls = {0.0, 0.1, 0.2};
    return ls;
}

utils::vec PreCompute::calc_new_vel(utils::vec vel_at_cop, double gamma_){
    return vel_at_cop;
}

void PreCompute::update_full_model(pybind11::list py_list, std::string shape_name){
    utils::properties properties2; 
    properties2.grid_shape = {pybind11::cast<int>(py_list[0]), pybind11::cast<int>(py_list[1])};
    properties2.grid_size = pybind11::cast<double>(py_list[2]);
    properties2.mu_c = 1.0;
    properties2.mu_s = 1.0;
    properties2.v_s = pybind11::cast<double>(py_list[5]);
    properties2.alpha = pybind11::cast<double>(py_list[6]);
    properties2.s0 = 100000.0;
    properties2.s1 = 0.0;
    properties2.s2 = 0.0;
    properties2.dt = pybind11::cast<double>(py_list[10]);
    properties2.z_ba_ratio = pybind11::cast<double>(py_list[11]);
    properties2.stability = false;
    properties2.elasto_plastic = false;
    properties2.steady_state = true;

    full_model.init_cpp(properties2, shape_name, 1.0);
}

void PreCompute::pre_comp_ls(int nr_segments_){
    nr_segments = nr_segments_;
    std::vector<double> f_t;
    std::vector<double> f_tau; 
    double f_t_max; 
    double f_tau_max;

    cop = full_model.get_cop();

    f_t = full_model.step_cpp({1.0, 0.0, 0.0});

    f_t_max = abs(f_t[0]);
    f_tau = full_model.step_cpp({0.0, 0.0, 1.0});
    f_tau_max = abs(f_tau[2]);

    gamma = calc_gamma();

    std::vector<utils::four_points> ls_list_temp(4*nr_segments*nr_segments);
    ls_list = ls_list_temp;

    for (int r = 0; r < 4*nr_segments; r++){
        for (int i = 0; i < nr_segments; i++){
            calc_4_points(r, i, f_t_max, f_tau_max);
        }
    }


}

double PreCompute::calc_gamma(){
    std::vector<double> cop_;
    std::vector<double> f;
    utils::properties p;
    utils::vec vel;
    utils::vec vel_cop;
    double gamma;

    p = full_model.get_properties();

    cop_ = full_model.get_cop();

    vel.x = 0.0; vel.y = 0.0; vel.tau = 1.0;
    vel_cop = utils::vel_to_point(utils::negate_vector(cop_), vel);
    f = full_model.step_cpp(vel_cop);
    f = full_model.get_force_at_cop();
    gamma = abs(f[2])/(p.mu_c);

    return gamma;
}

std::vector<double> PreCompute::update_viscus_scale(){
    std::vector<double> vis_scale = {0.0, 0.1, 0.2};
    return vis_scale;
}


void ReducedFrictionModel::init(pybind11::list py_list, std::string shape_name, double fn){

    properties.grid_shape = {pybind11::cast<int>(py_list[0]), pybind11::cast<int>(py_list[1])};
    properties.grid_size = pybind11::cast<double>(py_list[2]);
    properties.mu_c = pybind11::cast<double>(py_list[3]);
    properties.mu_s = pybind11::cast<double>(py_list[4]);
    properties.v_s = pybind11::cast<double>(py_list[5]);
    properties.alpha = pybind11::cast<double>(py_list[6]);
    properties.s0 = pybind11::cast<double>(py_list[7]);
    properties.s1 = pybind11::cast<double>(py_list[8]);
    properties.s2 = pybind11::cast<double>(py_list[9]);
    properties.dt = pybind11::cast<double>(py_list[10]);
    properties.z_ba_ratio = pybind11::cast<double>(py_list[11]);
    properties.stability = pybind11::cast<bool>(py_list[12]);
    properties.elasto_plastic = pybind11::cast<bool>(py_list[13]);
    properties.steady_state = pybind11::cast<bool>(py_list[14]);
    // update p_x_y
    p_x_y.init(shape_name, properties.grid_size, properties.grid_shape, fn);
    // update pre-compute
    pre_compute.update_full_model(py_list, shape_name);
    pre_compute.pre_comp_ls(20);
    gamma = pre_compute.calc_gamma();
    viscus_scale = pre_compute.update_viscus_scale();
    std::vector<double> lugre_f(3, 0.0);
    lugre.f = lugre_f;

    std::vector<double> lugre_z(3, 0.0);
    lugre.z = lugre_z;

}



std::vector<double> ReducedFrictionModel::step(pybind11::list py_list){
    std::vector<double> f = {0.0, 0.1, 0.2};
    return f;
}


namespace py = pybind11;

//FullFrictionModel hh;
//int ss = hh.step(0.1);

PYBIND11_MODULE(ReducedFrictionModelCPPClass, var) {
    var.doc() = "pybind11 example module for a class";
    
    py::class_<ReducedFrictionModel>(var, "ReducedFrictionModel")
        .def(py::init<>())
        .def("init", &ReducedFrictionModel::init)
        .def("step", &ReducedFrictionModel::step);
}

