# include "reducedModel.h"
#include <iostream>
#include <algorithm>
#include <cmath>

utils::closest_sample PreCompute::get_closest_samples(utils::vec vel, double gamma_){

    double a1 = std::atan2(vel.y, vel.x);

    if (vel.tau < 0){
        if (a1 < M_PI){
            a1 += M_PI;
        }else{
            a1 += -M_PI;
        }
    }

    if (a1 < 0){
        a1 += 2.0*M_PI;
    }

    a1 *= 2*nr_segments/M_PI;

    double v_xy_norm = std::sqrt(pow(vel.x, 2) + pow(vel.y, 2));
    v_xy_norm = gamma_ * v_xy_norm /(pow(gamma, 2));

    double a2 = std::atan2(v_xy_norm, abs(vel.tau)) * 2 * nr_segments / M_PI;

    int r1 = int(a1);
    int i1 = int(a2);
    double dr = a1 - r1;
    double di = a2 - i1;

    if (i1 >= nr_segments){
        i1 = nr_segments - 1;
        di = 1.0;
    }
    if (r1 >= 4*nr_segments){
        r1 = 4*nr_segments - 1;
        dr = 1.0;
    }

    int idx = calc_idx(r1, i1);

    utils::closest_sample sample;

    sample.force = ls_list[idx];
    sample.dr = dr;
    sample.di = di;
    return sample;
} 

void PreCompute::pre_compute_pos(){
    utils::vec vel;
    vel.x = 0.0; vel.y = 0.0; vel.tau=1.0;
    std::vector<double> ls_;
    ls_ = get_bilinear_interpolation(vel, gamma);

    double ax = std::atan2(ls_[2], ls_[0]) + M_PI/2.0;
    double ay = std::atan2(ls_[2], ls_[1]) + M_PI/2.0;

    int itter = 5;
    double rx = ax;
    double ry = ay;

    double bx;
    double by;

    for (int i = 0; i < itter; i++){
        vel.x = gamma * std::sin(rx);
        vel.y = gamma * std::sin(ry);

        ls_ = get_bilinear_interpolation(vel, gamma);

        double bx = std::atan2(ls_[2], ls_[0]) + M_PI/2.0 + ax;
        double by = std::atan2(ls_[2], ls_[1]) + M_PI/2.0 + ay;

        if (ax != 0){
            rx = rx * bx / ax;
        }   
        if (ay != 0){
            ry = ry * by / ay;
        }
    }
    
    std::vector<double> pos_(2);
    pos_[0] = - std::sin(ry);
    pos_[1] = std::sin(rx);
    pos = pos_;
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
    utils::closest_sample sample;
    sample = get_closest_samples(vel, gamma_);
    double s1 = (1-sample.dr)*(1-sample.di);
    double s2 = sample.dr*(1-sample.di);
    double s3 = sample.di*(1-sample.dr);
    double s4 = sample.dr*sample.di;

    double ls_x = s1*sample.force.f1[0] + s2*sample.force.f2[0] + s3*sample.force.f3[0] + s4*sample.force.f4[0];
    double ls_y = s1*sample.force.f1[1] + s2*sample.force.f2[1] + s3*sample.force.f3[1] + s4*sample.force.f4[1];
    double ls_tau = s1*sample.force.f1[2] + s2*sample.force.f2[2] + s3*sample.force.f3[2] + s4*sample.force.f4[2];
    std::vector<double> ls = {ls_x, ls_y, ls_tau};
    return ls;
}

utils::vec PreCompute::calc_new_vel(utils::vec vel_at_cop, double gamma_){

    std::vector<double> pos_(2, 0.0);
    pos_[0] = gamma_*pos[0];
    pos_[1] = gamma_*pos[1];
    utils::vec vel;

    vel = utils::vel_to_point(pos_, vel_at_cop);

    double p_xy_n = std::sqrt(pow(pos_[0], 2) + pow(pos_[1], 2));

    double r = 0;
    if (p_xy_n != 0){
        double p_x_n = std::abs(pos_[0])/p_xy_n;
        double p_y_n = std::abs(pos_[1])/p_xy_n;
        double nn = std::sqrt(pow(vel_at_cop.x * p_x_n, 2) + pow(vel_at_cop.y * p_y_n, 2)) / gamma_;
        r = (2 * std::atan2(std::abs(vel_at_cop.tau), nn))/M_PI;
    }
    utils::vec new_vel;
    new_vel.x = r * vel.x + (1-r) * vel_at_cop.x;
    new_vel.y = r * vel.y + (1-r) * vel_at_cop.y;
    new_vel.tau = r * vel.tau + (1-r) * vel_at_cop.tau;
    return new_vel;
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

    utils::properties properties3; 
    properties3.grid_shape = {pybind11::cast<int>(py_list[0]), pybind11::cast<int>(py_list[1])};
    properties3.grid_size = pybind11::cast<double>(py_list[2]);
    properties3.mu_c = 0.0;
    properties3.mu_s = 0.0;
    properties3.v_s = pybind11::cast<double>(py_list[5]);
    properties3.alpha = pybind11::cast<double>(py_list[6]);
    properties3.s0 = 100000.0;
    properties3.s1 = 0.0;
    properties3.s2 = 1.0;
    properties3.dt = pybind11::cast<double>(py_list[10]);
    properties3.z_ba_ratio = pybind11::cast<double>(py_list[11]);
    properties3.stability = false;
    properties3.elasto_plastic = false;
    properties3.steady_state = true;

    full_model_viscus.init_cpp(properties3, shape_name, 1.0);
}

void PreCompute::pre_comp_ls(int nr_segments_){
    nr_segments = nr_segments_;
    std::vector<double> f_t;
    std::vector<double> f_tau; 
    double f_t_max; 
    double f_tau_max;

    cop = full_model.get_cop();

    f_t = full_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), {1.0, 0.0, 0.0}));

    f_t = full_model.get_force_at_cop();
    f_t_max = abs(f_t[0]);
    f_tau = full_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), {0.0, 0.0, 1.0}));

    f_tau =  full_model.get_force_at_cop();
    f_tau_max = abs(f_tau[2]);

    gamma = calc_gamma();

    std::vector<utils::four_points> ls_list_temp(4*nr_segments*nr_segments);
    ls_list = ls_list_temp;

    for (int r = 0; r < 4*nr_segments; r++){
        for (int i = 0; i < nr_segments; i++){
            calc_4_points(r, i, f_t_max, f_tau_max);
        }
    }
    update_viscus_scale();
    pre_compute_pos();


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

void PreCompute::update_viscus_scale(){
    utils::vec vel;
    vel.x = 0.0; vel.y = 0.0; vel.tau = 1.0;
    std::vector<double> cop_ = full_model_viscus.get_cop();
    tau_visc = full_model_viscus.step_cpp(utils::vel_to_point(utils::negate_vector(cop_), vel))[2];
}

std::vector<double> PreCompute::get_viscus_scale(double gamma_){
    std::vector<double> visc_scale(3, 1.0);
    double m = pow(gamma_, 2);
    visc_scale[2] = std::abs(tau_visc/m);
    return visc_scale;
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
    double N_LS = pybind11::cast<int>(py_list[15]);

    // update p_x_y
    p_x_y.init(shape_name, properties.grid_size, properties.grid_shape, fn);
    // update pre-compute
    pre_compute.update_full_model(py_list, shape_name);
    pre_compute.pre_comp_ls(N_LS);
    gamma = pre_compute.calc_gamma();
    viscus_scale = pre_compute.get_viscus_scale(gamma);
    std::vector<double> lugre_f(3, 0.0);
    lugre.f = lugre_f;

    std::vector<double> lugre_z(3, 0.0);
    lugre.z = lugre_z;

}



std::vector<double> ReducedFrictionModel::step(pybind11::list py_list){
    velocity.x = pybind11::cast<double>(py_list[0]);
    velocity.y = pybind11::cast<double>(py_list[1]);
    velocity.tau = pybind11::cast<double>(py_list[2]);
    
    shape_info_var_red = p_x_y.get_red(properties.grid_size);
    
    utils::vec vel_cop = utils::vel_to_point(shape_info_var_red.cop, velocity);

    update_lugre(vel_cop);

    force_vec_cop = lugre.f;

    std::vector<double> force_at_center;

    force_at_center = move_force_to_center(force_vec_cop);

    return force_at_center;
}


void ReducedFrictionModel::update_lugre(utils::vec vel_cop){
    utils::properties p = properties;
    double v_norm = 0;
    double alpha;
    //std::vector<double> beta;
    std::vector<double> Av = {0.0, 0.0, 0.0};
    std::vector<double> SAv = {0.0, 0.0, 0.0};
    std::vector<double> z_ss = {0.0, 0.0, 0.0}; // [x, y, tau]
    std::vector<double> dz = {0.0, 0.0, 0.0}; // [x, y, tau]
    std::vector<double> delta_z = {0.0, 0.0, 0.0}; // [x, y, tau]
    calc_SAv(vel_cop, v_norm, Av, SAv);
    double g = p.mu_c + (p.mu_s - p.mu_c) * exp(- pow((v_norm/p.v_s), p.alpha));
    if (v_norm != 0){
        z_ss[0] = SAv[0]*g/(p.s0*v_norm);
        z_ss[1] = SAv[1]*g/(p.s0*v_norm);
        z_ss[2] = SAv[2]*g/(p.s0*v_norm);
       
    }else{
        z_ss[0] = 0.0;
        z_ss[1] = 0.0;
        z_ss[2] = 0.0;
    }

    if (p.steady_state == true){
        lugre.f[0] = -(p.s0 * z_ss[0] + viscus_scale[0] * p.s2*Av[0]) * shape_info_var_red.fn;
        lugre.f[1] = -(p.s0 * z_ss[1] + viscus_scale[1] * p.s2*Av[1]) * shape_info_var_red.fn;
        lugre.f[2] = -(p.s0 * z_ss[2] + viscus_scale[2] * p.s2*Av[2]) * shape_info_var_red.fn;
        return;
    }

    if (p.elasto_plastic == true){
        std::vector<double> z_ = {0.0, 0.0, 0.0};
        std::vector<double> z_ss_ = {0.0, 0.0, 0.0};
        std::vector<double> v_ = {0.0, 0.0, 0.0};
        z_ = lugre.z;
        z_[2] = z_[2]*(1/gamma);
        z_ss_ = z_ss;
        z_ss_[2] =  z_ss_[2]*(1/gamma);
        v_ = SAv;
        v_[2] = v_[2] *(1/gamma);
        alpha = utils::elasto_plastic(z_, z_ss_, p.z_ba_ratio, v_, 3);
    }else{
        alpha = 1;
    }
    dz[0] = SAv[0] - alpha * lugre.z[0] * p.s0 * v_norm / g;
    dz[1] = SAv[1] - alpha * lugre.z[1] * p.s0 * v_norm / g;
    dz[2] = SAv[2] - alpha * lugre.z[2] * p.s0 * v_norm / g;

    
    if (p.stability == true){
        delta_z[0] = (z_ss[0] - lugre.z[0]) / p.dt;
        delta_z[1] = (z_ss[1] - lugre.z[1]) / p.dt;
        delta_z[2] = (z_ss[2] - lugre.z[2]) / p.dt;

        dz[0] = std::min(abs(dz[0]), abs(delta_z[0]))*((dz[0] > 0) - (dz[0] < 0));
        dz[1] = std::min(abs(dz[1]), abs(delta_z[1]))*((dz[1] > 0) - (dz[1] < 0));
        dz[2] = std::min(abs(dz[2]), abs(delta_z[2]))*((dz[2] > 0) - (dz[2] < 0));
    }

    lugre.z[0] += dz[0] * p.dt;
    lugre.z[1] += dz[1] * p.dt;
    lugre.z[2] += dz[2] * p.dt;

    lugre.f[0] = -(p.s0 * lugre.z[0] + p.s1 * dz[0] + viscus_scale[0]*p.s2*Av[0]) * shape_info_var_red.fn;
    lugre.f[1] = -(p.s0 * lugre.z[1] + p.s1 * dz[1] + viscus_scale[1]*p.s2*Av[1]) * shape_info_var_red.fn;
    lugre.f[2] = -(p.s0 * lugre.z[2] + p.s1 * dz[2] + viscus_scale[2]*p.s2*Av[2]) * shape_info_var_red.fn;

}   


std::vector<double> ReducedFrictionModel::move_force_to_center(std::vector<double> force_at_cop){
    std::vector<double> f_t(3, 0.0);
    std::vector<double> cop3(3, 0.0);
    std::vector<double> m;
    std::vector<double> force_at_center(3);

    f_t[0] = force_at_cop[0]; f_t[1] = force_at_cop[1];
    cop3[0] = shape_info_var_red.cop[0]; cop3[1] = shape_info_var_red.cop[1];
    m = utils::crossProduct(cop3, f_t);
    force_at_center[0] = force_at_cop[0];
    force_at_center[1] = force_at_cop[1];
    force_at_center[2] = force_at_cop[2] + m[2];
     
    return force_at_center;
}


std::vector<double> ReducedFrictionModel::calc_beta(utils::vec vel_cop, std::vector<double>& vel_cop_tau, double& v_norm){
    std::vector<double> beta(3, 0.0);
    std::vector<double> ls;
    utils::vec new_vel;

    ls = pre_compute.get_bilinear_interpolation(vel_cop, gamma);
    new_vel = pre_compute.calc_new_vel(vel_cop, gamma);
    vel_cop_tau[0] = new_vel.x;
    vel_cop_tau[1] = new_vel.y;
    vel_cop_tau[2] = new_vel.tau * gamma;
    v_norm = std::sqrt(pow(vel_cop_tau[0], 2) + pow(vel_cop_tau[1], 2) + pow(vel_cop_tau[2], 2));

    for (int i = 0; i < 3; i++){
        if (vel_cop_tau[i] != 0){
            beta[i] = abs(ls[i]) *v_norm / abs(vel_cop_tau[i]);
        }else{
            beta[i] = 1;
        }
    }

    return beta;
}


void ReducedFrictionModel::calc_SAv(utils::vec vel_cop, double& v_norm, std::vector<double>& Av, std::vector<double>& SAv){
    std::vector<double> beta(3, 0.0);
    std::vector<double> ls;
    utils::vec new_vel;
    double sx; double sy;
    ls = pre_compute.get_bilinear_interpolation(vel_cop, gamma);
    new_vel = pre_compute.calc_new_vel(vel_cop, gamma);

    Av[0] = new_vel.x;
    Av[1] = new_vel.y;
    Av[2] = gamma*gamma*vel_cop.tau;

    v_norm = std::sqrt(vel_cop.x*Av[0]+ vel_cop.y*Av[1] + vel_cop.tau*Av[2]);

    int signx = ( Av[0] < 0) ? -1 : 1;
    int signy = ( Av[1] < 0) ? -1 : 1;
    int signt = ( Av[2] < 0) ? -1 : 1;


    SAv[0] = abs(ls[0]) *signx* v_norm;
    SAv[1] = abs(ls[1]) *signy* v_norm;
    SAv[2] = abs(gamma*ls[2]) *signt* v_norm;
    //std::cout <<  SAv[0] << " " <<  SAv[1] << " " <<  SAv[2] << " Qv " << Av[0]<< " "<< Av[1]<< " "<< Av[2]<< " vnorm " << v_norm << std::endl;

}


std::vector<double> ReducedFrictionModel::get_force_at_cop(){
    return force_vec_cop;
}


namespace py = pybind11;

//FullFrictionModel hh;
//int ss = hh.step(0.1);

PYBIND11_MODULE(ReducedFrictionModelCPPClass, var) {
    var.doc() = "pybind11 example module for a class";
    
    py::class_<ReducedFrictionModel>(var, "ReducedFrictionModel")
        .def(py::init<>())
        .def("init", &ReducedFrictionModel::init)
        .def("step", &ReducedFrictionModel::step)
        .def("get_force_at_cop", &ReducedFrictionModel::get_force_at_cop);
}

