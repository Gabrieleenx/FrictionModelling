# include "reducedModel.h"
#include <iostream>
#include <algorithm>
#include <cmath>

utils::closest_sample PreCompute::get_closest_samples(utils::vec vel, double ra_){
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
    //v_xy_norm = ra_ * v_xy_norm /(pow(ra, 2));
    double a2 = std::atan2(v_xy_norm,  ra_ * abs(vel.tau)) * 2 * nr_segments / M_PI;

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
    double omega = 1.0;
    vel.x = 0.0; vel.y = 0.0; vel.tau=omega;
    std::vector<double> ls_;
    ls_ = get_bilinear_interpolation(vel, ra);
    double vx0 = ra*omega*ls_[0];
    double vy0 = ra*omega*ls_[1];
    double vx = vx0;
    double vy = vy0;
    while (sqrt(ls_[0]*ls_[0] + ls_[1]+ls_[1]) > 1e-6) {
        vel.x = vx; vel.y = vy;
        ls_ = get_bilinear_interpolation(vel, ra);
        if (vx0 != 0){
            vx = vx*(ra * omega * ls_[0] + vx0)/vx0;
        }
        if (vy0 != 0){
            vy = vy*(ra * omega * ls_[1] + vy0)/vy0;
        }
                

    }
        
    std::vector<double> pos_(2);
    pos_[0] = -vy/omega;
    pos_[1] = vx/omega;
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
    v = ra * std::sqrt(1 - pow(vel.tau, 2));
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
            distributed_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), vel));
            fj = normalize_force(distributed_model.get_force_at_cop(), f_t_max, f_tau_max);    
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

std::vector<double> PreCompute::get_bilinear_interpolation(utils::vec vel, double ra_){
    utils::closest_sample sample;
    sample = get_closest_samples(vel, ra_);
    int sign_p = ( vel.tau < 0) ? -1 : 1;
    double s1 = (1-sample.dr)*(1-sample.di);
    double s2 = sample.dr*(1-sample.di);
    double s3 = sample.di*(1-sample.dr);
    double s4 = sample.dr*sample.di;

    double ls_x = s1*sample.force.f1[0] + s2*sample.force.f2[0] + s3*sample.force.f3[0] + s4*sample.force.f4[0];
    double ls_y = s1*sample.force.f1[1] + s2*sample.force.f2[1] + s3*sample.force.f3[1] + s4*sample.force.f4[1];
    double ls_tau = s1*sample.force.f1[2] + s2*sample.force.f2[2] + s3*sample.force.f3[2] + s4*sample.force.f4[2];
    std::vector<double> ls = {sign_p*ls_x, sign_p*ls_y, sign_p*ls_tau};
    return ls;
}

std::vector<double> PreCompute::calc_skew_var(utils::vec vel_at_cop, double ra_){

    double p_xy_n = std::sqrt(pow(pos[0], 2) + pow(pos[1], 2));
    double p_x_n = 0;
    double p_y_n = 0;

    if (p_xy_n != 0){
        p_x_n = std::abs(pos[0])/p_xy_n;
        p_y_n = std::abs(pos[1])/p_xy_n;
    }
    double nn = std::sqrt(pow(vel_at_cop.x * p_x_n, 2) + pow(vel_at_cop.y * p_y_n, 2)) / ra_;
    double s_n = (2 * std::atan2(std::abs(vel_at_cop.tau), nn))/M_PI;

    double sx = -ra_/ra * s_n * pos[1];
    double sy = ra_/ra * s_n * pos[0];

    std::vector<double> s_x_y(2, 0.0);
    s_x_y[0] = sx;
    s_x_y[1] = sy;
    
    return s_x_y;
}


void PreCompute::update_distributed_model(pybind11::list py_list, std::string shape_name){
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

    distributed_model.init_cpp(properties2, shape_name, 1.0);

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

    distributed_model_viscus.init_cpp(properties3, shape_name, 1.0);
}

void PreCompute::update_surface(std::vector<double> new_surf, std::string shape_name, double fn){
    distributed_model.update_surface_cpp(new_surf, shape_name, fn);
    distributed_model_viscus.update_surface_cpp(new_surf, shape_name, fn);
}

void PreCompute::pre_comp_ls(int nr_segments_){
    nr_segments = nr_segments_;
    std::vector<double> f_t;
    std::vector<double> f_tau; 
    double f_t_max; 
    double f_tau_max;

    cop = distributed_model.get_cop();

    f_t = distributed_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), {1.0, 0.0, 0.0}));

    f_t = distributed_model.get_force_at_cop();
    f_t_max = abs(f_t[0]);
    f_tau = distributed_model.step_cpp(utils::vel_to_point(utils::negate_vector(cop), {0.0, 0.0, 1.0}));

    f_tau =  distributed_model.get_force_at_cop();
    f_tau_max = abs(f_tau[2]);

    ra = calc_ra();

    std::vector<utils::four_points> ls_list_temp(4*nr_segments*nr_segments);
    ls_list = ls_list_temp;

    for (int r = 0; r < 4*nr_segments; r++){
        for (int i = 0; i < nr_segments; i++){
            calc_4_points(r, i, f_t_max, f_tau_max);
        }
    }
    update_viscus_scale();
    //pre_compute_pos();


}

double PreCompute::calc_ra(){
    std::vector<double> cop_;
    std::vector<double> f;
    utils::properties p;
    utils::vec vel;
    utils::vec vel_cop;
    double ra;

    p = distributed_model.get_properties();

    cop_ = distributed_model.get_cop(); 
    vel.x = 0.0; vel.y = 0.0; vel.tau = 1.0;
    vel_cop = utils::vel_to_point(utils::negate_vector(cop_), vel);
    f = distributed_model.step_cpp(vel_cop);
    f = distributed_model.get_force_at_cop();

    ra = abs(f[2])/(p.mu_c);

    return ra;
}

void PreCompute::update_viscus_scale(){
    utils::vec vel;
    vel.x = 0.0; vel.y = 0.0; vel.tau = 1.0;
    std::vector<double> cop_ = distributed_model_viscus.get_cop();
    tau_visc = distributed_model_viscus.step_cpp(utils::vel_to_point(utils::negate_vector(cop_), vel))[2];
}

std::vector<double> PreCompute::get_viscus_scale(double ra_){
    std::vector<double> visc_scale(3, 1.0);
    double m = pow(ra_, 2);
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
    N_LS = pybind11::cast<int>(py_list[15]);
    // update p_x_y
    p_x_y.init(shape_name, properties.grid_size, properties.grid_shape, fn);

    // update pre-compute
    pre_compute.update_distributed_model(py_list, shape_name);
    pre_compute.pre_comp_ls(N_LS);
    ra = pre_compute.calc_ra();
    viscus_scale = pre_compute.get_viscus_scale(ra);
    std::vector<double> lugre_f(3, 0.0);
    lugre.f = lugre_f;

    std::vector<double> lugre_z(3, 0.0);
    lugre.z = lugre_z;

    std::vector<double> lugre_dz(3, 0.0);
    lugre.dz = lugre_dz;

}

void ReducedFrictionModel::update_surface(pybind11::list py_list, std::string shape_name, double fn, int recalc){
    std::vector<double> new_surf(properties.grid_shape[0]*properties.grid_shape[1], 0.0);
    new_surf = pybind11::cast<std::vector<double>>(py_list);
    p_x_y.update_shape_info(shape_name, new_surf);
    p_x_y.set_fn(fn);

    // update pre-compute
    pre_compute.update_surface(new_surf, shape_name, fn);
    if(recalc == 1){
        pre_compute.pre_comp_ls(N_LS);
    }
    ra = pre_compute.calc_ra();
    viscus_scale = pre_compute.get_viscus_scale(ra);
}



std::vector<double> ReducedFrictionModel::step(pybind11::list py_list){
    velocity.x = pybind11::cast<double>(py_list[0]);
    velocity.y = pybind11::cast<double>(py_list[1]);
    velocity.tau = pybind11::cast<double>(py_list[2]);
    
    shape_info_var_red = p_x_y.get_red(properties.grid_size);
    
    utils::vec vel_cop = utils::vel_to_point(shape_info_var_red.cop, velocity);

    update_lugre(vel_cop);

    force_vec_cop = lugre.f;

    //std::vector<double> force_at_center;

    force_at_center = move_force_to_center(force_vec_cop);

    return force_at_center;
}


void ReducedFrictionModel::update_lugre(utils::vec vel_cop){
    utils::properties p = properties;
    double v_n = 0;
    double beta;
    std::vector<double> Av = {0.0, 0.0, 0.0};
    std::vector<double> w = {0.0, 0.0, 0.0};
    std::vector<double> z_ss = {0.0, 0.0, 0.0}; // [x, y, tau]
    std::vector<double> dz = {0.0, 0.0, 0.0}; // [x, y, tau]
    std::vector<double> delta_z = {0.0, 0.0, 0.0}; // [x, y, tau]
    calc_w(vel_cop, v_n, Av, w);

    double g = p.mu_c + (p.mu_s - p.mu_c) * exp(- pow((v_n/p.v_s), p.alpha));
    if (v_n != 0){
        z_ss[0] = w[0]*g/(p.s0);
        z_ss[1] = w[1]*g/(p.s0);
        z_ss[2] = w[2]*g/(p.s0);
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
        z_[2] = z_[2]*(1/ra);
        z_ss_ = z_ss;
        z_ss_[2] =  z_ss_[2]*(1/ra);
        //v_ = SAv;
        v_[0] = w[0]*v_n;
        v_[1] =w[1]*v_n;
        v_[2] = w[2]*v_n*(1/ra);
        beta = utils::elasto_plastic(z_, z_ss_, p.z_ba_ratio, v_, 3);
        lugre.beta = beta;
    }else{
        beta = 1;
        lugre.beta = beta;
    }

    dz[0] = (w[0] - beta * lugre.z[0] * p.s0 / g)*v_n;
    dz[1] = (w[1] - beta * lugre.z[1] * p.s0 / g)*v_n;
    dz[2] = (w[2] - beta * lugre.z[2] * p.s0 / g)*v_n;
    
    if (p.stability == true){
        delta_z[0] = (z_ss[0] - lugre.z[0]) / p.dt;
        delta_z[1] = (z_ss[1] - lugre.z[1]) / p.dt;
        delta_z[2] = (z_ss[2] - lugre.z[2]) / p.dt;

        dz[0] = std::min(abs(dz[0]), abs(delta_z[0]))*((dz[0] > 0) - (dz[0] < 0));
        dz[1] = std::min(abs(dz[1]), abs(delta_z[1]))*((dz[1] > 0) - (dz[1] < 0));
        dz[2] = std::min(abs(dz[2]), abs(delta_z[2]))*((dz[2] > 0) - (dz[2] < 0));
    }

    lugre.f[0] = -(p.s0 * lugre.z[0] + p.s1 * dz[0] + viscus_scale[0]*p.s2*Av[0]) * shape_info_var_red.fn;
    lugre.f[1] = -(p.s0 * lugre.z[1] + p.s1 * dz[1] + viscus_scale[1]*p.s2*Av[1]) * shape_info_var_red.fn;
    lugre.f[2] = -(p.s0 * lugre.z[2] + p.s1 * dz[2] + viscus_scale[2]*p.s2*Av[2]) * shape_info_var_red.fn;
    
    lugre.dz[0] = dz[0];
    lugre.dz[1] = dz[1];
    lugre.dz[2] = dz[2];

    lugre.z[0] += dz[0] * p.dt;
    lugre.z[1] += dz[1] * p.dt;
    lugre.z[2] += dz[2] * p.dt;
}   


std::vector<double> ReducedFrictionModel::move_force_to_center(std::vector<double> force_at_cop){
    std::vector<double> f_t(3, 0.0);
    std::vector<double> cop3(3, 0.0);
    std::vector<double> m;
    std::vector<double> force_at_center_(3);

    f_t[0] = force_at_cop[0]; f_t[1] = force_at_cop[1];
    cop3[0] = shape_info_var_red.cop[0]; cop3[1] = shape_info_var_red.cop[1];
    m = utils::crossProduct(cop3, f_t);
    force_at_center_[0] = force_at_cop[0];
    force_at_center_[1] = force_at_cop[1];
    force_at_center_[2] = force_at_cop[2] + m[2];
     
    return force_at_center_;
}


void ReducedFrictionModel::calc_w(utils::vec vel_cop, double& v_n, std::vector<double>& Av, std::vector<double>& w){
    std::vector<double> beta(3, 0.0);
    std::vector<double> ls;    
    //std::vector<double> s_x_y(2, 0.0);

    utils::vec new_vel;
    //double sx; double sy;
    double vx = vel_cop.x; double vy = vel_cop.y; double vt = vel_cop.tau; 
    ls = pre_compute.get_bilinear_interpolation(vel_cop, ra);
    //s_x_y = pre_compute.calc_skew_var(vel_cop, ra);

    //shape_info_var_red.s_x_y = s_x_y;
    //sx =s_x_y[0];
    //sy =s_x_y[1];
    
    Av[0] = vx ; //+ vt*sx;
    Av[1] = vy; // + vt*sy;
    Av[2] = ra*ra*vt;

    //v_n = std::sqrt(vx*(vx + vt*sx) + vy*(vy + vt*sy) + vt*vt*ra*ra);
    v_n = std::sqrt(vx*(vx) + vy*(vy) + vt*vt*ra*ra);

    //int signx = ( Av[0] < 0) ? -1 : 1;
    //int signy = ( Av[1] < 0) ? -1 : 1;
    //int signt = ( Av[2] < 0) ? -1 : 1;

    w[0] = -ls[0]; //abs(ls[0]) *signx;
    w[1] = -ls[1]; //abs(ls[1]) *signy;
    w[2] = -ra*ls[2]; //abs(ra*ls[2]) *signt;

}


std::vector<double> ReducedFrictionModel::step_ode(pybind11::list py_y, pybind11::list py_vel){
    velocity.x = pybind11::cast<double>(py_vel[0]);
    velocity.y = pybind11::cast<double>(py_vel[1]);
    velocity.tau = pybind11::cast<double>(py_vel[2]);
    
    shape_info_var_red = p_x_y.get_red(properties.grid_size);

    utils::vec vel_cop = utils::vel_to_point(shape_info_var_red.cop, velocity);

    lugre.z[0] = pybind11::cast<double>(py_y[0]);
    lugre.z[1] = pybind11::cast<double>(py_y[1]);
    lugre.z[2] = pybind11::cast<double>(py_y[2]);

    update_lugre(vel_cop);

    force_vec_cop = lugre.f;

    force_at_center = move_force_to_center(force_vec_cop);

    return lugre.dz;

}


std::vector<double> ReducedFrictionModel::get_force_at_cop(){
    return force_vec_cop;
}


std::vector<double> ReducedFrictionModel::get_force_at_center(){
    return force_at_center;
}

std::vector<double> ReducedFrictionModel::get_cop(){
    shape_info_var_red = p_x_y.get_red(properties.grid_size);
    return shape_info_var_red.cop;
}

namespace py = pybind11;



PYBIND11_MODULE(ReducedFrictionModelCPPClass, var) {
    var.doc() = "pybind11 example module for a class";
    
    py::class_<ReducedFrictionModel>(var, "ReducedFrictionModel")
        .def(py::init<>())
        .def("init", &ReducedFrictionModel::init)
        .def("step", &ReducedFrictionModel::step)
        .def("step_ode", &ReducedFrictionModel::step_ode)
        .def("get_force_at_center", &ReducedFrictionModel::get_force_at_center)
        .def("get_force_at_cop", &ReducedFrictionModel::get_force_at_cop)
        .def("get_cop", &ReducedFrictionModel::get_cop)
        .def("update_surface", &ReducedFrictionModel::update_surface)
        .def("set_fn", &ReducedFrictionModel::set_fn);
}

