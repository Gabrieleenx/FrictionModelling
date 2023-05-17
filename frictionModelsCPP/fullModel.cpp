#include "fullModel.h"
#include <iostream>
#include <algorithm>

void FullFrictionModel::init(pybind11::list py_list, std::string shape_name, double fn){
    
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

    p_x_y.init(shape_name, properties.grid_size, properties.grid_shape, fn);

    int n_x = properties.grid_shape[0];
    std::vector<double> position_vec_x_new(n_x, 0.0);
    for (int ix = 0; ix<n_x; ix++){
        position_vec_x_new[ix] = (ix + 0.5 - n_x/2.0)*properties.grid_size;
    }
    position_vec_x.assign(position_vec_x_new.begin(), position_vec_x_new.end());

    int n_y = properties.grid_shape[1];
    std::vector<double> position_vec_y_new(n_y, 0.0);
    for (int iy = 0; iy<n_y; iy++){
        position_vec_y_new[iy] = (iy + 0.5 - n_y/2.0)*properties.grid_size;
    }
    position_vec_y.assign(position_vec_y_new.begin(), position_vec_y_new.end());

    std::vector<std::vector<std::vector<double>>> lugre_f(properties.grid_shape[0], std::vector<std::vector<double>>(properties.grid_shape[1], std::vector<double>(int(2), 0.0)));
    lugre.f = lugre_f;

    std::vector<std::vector<std::vector<double>>> lugre_z(properties.grid_shape[0], std::vector<std::vector<double>>(properties.grid_shape[1], std::vector<double>(int(2), 0.0)));
    lugre.z = lugre_z;
}


void FullFrictionModel::init_cpp(utils::properties properties_, std::string shape_name, double fn){
    
    properties = properties_;

    p_x_y.init(shape_name, properties.grid_size, properties.grid_shape, fn);

    int n_x = properties.grid_shape[0];
    std::vector<double> position_vec_x_new(n_x, 0.0);
    for (int ix = 0; ix<n_x; ix++){
        position_vec_x_new[ix] = (ix + 0.5 - n_x/2.0)*properties.grid_size;
    }
    position_vec_x.assign(position_vec_x_new.begin(), position_vec_x_new.end());

    int n_y = properties.grid_shape[1];
    std::vector<double> position_vec_y_new(n_y, 0.0);
    for (int iy = 0; iy<n_y; iy++){
        position_vec_y_new[iy] = (iy + 0.5 - n_y/2.0)*properties.grid_size;
    }
    position_vec_y.assign(position_vec_y_new.begin(), position_vec_y_new.end());

    std::vector<std::vector<std::vector<double>>> lugre_f(properties.grid_shape[0], std::vector<std::vector<double>>(properties.grid_shape[1], std::vector<double>(int(2), 0.0)));
    lugre.f = lugre_f;

    std::vector<std::vector<std::vector<double>>> lugre_z(properties.grid_shape[0], std::vector<std::vector<double>>(properties.grid_shape[1], std::vector<double>(int(2), 0.0)));
    lugre.z = lugre_z;

    shape_info_var = p_x_y.get(properties.grid_size);

}

std::vector<double> FullFrictionModel::get_cop(){
    return shape_info_var.cop;
}


utils::properties FullFrictionModel::get_properties(){
    return properties;
}

std::vector<double> FullFrictionModel::get_force_at_cop(){
    return force_vec_cop;
}


std::vector<double> FullFrictionModel::step(pybind11::list py_list){
    velocity.x = pybind11::cast<double>(py_list[0]);
    velocity.y = pybind11::cast<double>(py_list[1]);
    velocity.tau = pybind11::cast<double>(py_list[2]);
    shape_info_var = p_x_y.get(properties.grid_size);

    if (properties.steady_state == false){
        return step_single_point();
    }else{
        return step_bilinear();
    }

}


std::vector<double> FullFrictionModel::step_cpp(utils::vec vel){
    velocity = vel;
    shape_info_var = p_x_y.get(properties.grid_size);

    if (properties.steady_state == false){
        return step_single_point();
    }else{
        return step_bilinear();
    }

}


std::vector<double> FullFrictionModel::step_single_point(){
    update_velocity_grid(velocity);
    update_lugre();
    std::vector<double> force_vec;
    force_vec = approximate_integral();
    force_vec_cop = move_force_to_cop(force_vec);
    return force_vec;
}


std::vector<double> FullFrictionModel::step_bilinear(){
    std::vector<double> force_at_center(3);

    if(velocity.tau == 0){
        force_at_center = step_single_point();
    }else{
        int nx = properties.grid_shape[0];
        int ny = properties.grid_shape[1];
        double x0 = nx* properties.grid_size/2;
        double y0 = ny* properties.grid_size/2;
        double cor_x = -velocity.y/velocity.tau;
        double cor_y = velocity.x/velocity.tau;

        double ix_ = nx*(cor_x + x0)/(2*x0);
        int ix = int(ix_);
        double dx = ix_ - ix;

        double iy_ = ny*(cor_y + y0)/(2*y0);
        int iy = int(iy_);
        double dy = iy_ - iy;
        

        if (ix >= 0 && ix < nx && iy >= 0 && iy < ny){
            std::vector<int> ox = {0, 1, 0, 1};
            std::vector<int> oy = {0, 0, 1, 1};
            std::vector<std::vector<double>> f(4, std::vector<double>(3, 0.0));
            std::vector<double> cor(2);
            utils::vec velocity_;
            utils::vec velocity_cop;
            velocity_cop.x = 0.0; velocity_cop.y = 0.0; velocity_cop.tau = velocity.tau;
            for (int i = 0; i < 4; i++){
                cor[0] = (double(ix + ox[i])/nx) * (2*std::abs(x0)) - x0;
                cor[1] = (double(iy + oy[i]) / ny) * (2 * std::abs(y0)) - y0;
                velocity_ = utils::vel_to_point(utils::negate_vector(cor), velocity_cop);
                update_velocity_grid(velocity_);
                update_lugre();
                f[i] = approximate_integral();
            }

            double s1= (1 - dx) * (1 - dy);
            double s2 = dx * (1 - dy);
            double s3 = dy * (1 - dx);
            double s4 = dx * dy;

            force_at_center[0] =  s1*f[0][0] + s2*f[1][0] + s3*f[2][0] + s4*f[3][0];
            force_at_center[1] =  s1*f[0][1] + s2*f[1][1] + s3*f[2][1] + s4*f[3][1];
            force_at_center[2] =  s1*f[0][2] + s2*f[1][2] + s3*f[2][2] + s4*f[3][2];

        }else{
            force_at_center = step_single_point();

        }
    }


    force_vec_cop = move_force_to_cop(force_at_center);
    return force_at_center;
}


void FullFrictionModel::update_velocity_grid(utils::vec vel){
    std::vector<double> w = {0.0, 0.0, vel.tau};
    std::vector<double> p = {0.0, 0.0, 0.0};
    std::vector<double> r = {0.0, 0.0, 0.0};

    std::vector<std::vector<std::vector<double>>> new_vel_grid(properties.grid_shape[0], std::vector<std::vector<double>>(properties.grid_shape[1], std::vector<double>(int(2), 0.0)));
    
    for (int ix = 0; ix<properties.grid_shape[0]; ix++){
        p[0] = position_vec_x[ix];
        for (int iy = 0; iy<properties.grid_shape[1]; iy++){
            p[1] = position_vec_y[iy];
            r = utils::crossProduct(w, p);
            new_vel_grid[ix][iy][0] = r[0] + vel.x;
            new_vel_grid[ix][iy][1] = r[1] + vel.y;
        }
    }
    
    vel_grid.assign(new_vel_grid.begin(), new_vel_grid.end());
}


void FullFrictionModel::update_lugre(){
    double v_norm;
    double v_norm1;
    double vx;
    double vy;
    double g;
    double alpha;
    std::vector<double> z_ss = {0.0, 0.0}; // [x, y]
    std::vector<double> dz = {0.0, 0.0}; // [x, y]
    std::vector<double> delta_z = {0.0, 0.0}; // [x, y]

    utils::properties p = properties;

    for (int ix = 0; ix<p.grid_shape[0]; ix++){
        for (int iy = 0; iy<p.grid_shape[1]; iy++){
            vx = vel_grid[ix][iy][0];
            vy = vel_grid[ix][iy][1];
            v_norm = sqrt(vx*vx + vy*vy);

            g = p.mu_c + (p.mu_s - p.mu_c) * exp(- pow((v_norm/p.v_s), p.alpha));
            if (v_norm == 0){
                v_norm1 = 1;
            }else{
                v_norm1 = v_norm;
            }

            z_ss[0] = vx*g/(p.s0*v_norm1);
            z_ss[1] = vy*g/(p.s0*v_norm1);
            if (p.steady_state == true){
                lugre.f[ix][iy][0] = (p.s0 * z_ss[0] + p.s2*vx) * shape_info_var.f_n_grid[ix][iy];
                lugre.f[ix][iy][1] = (p.s0 * z_ss[1] + p.s2*vy) * shape_info_var.f_n_grid[ix][iy];
                continue;
            }

            
            if (p.elasto_plastic == true){
                alpha = utils::elasto_plastic(lugre.z[ix][iy], z_ss, p.z_ba_ratio, vel_grid[ix][iy], 2);
                dz[0] = vx - alpha * lugre.z[ix][iy][0] * p.s0 * v_norm / g;
                dz[1] = vy - alpha * lugre.z[ix][iy][1] * p.s0 * v_norm / g;
            }else{
                dz[0] = vx - lugre.z[ix][iy][0] * p.s0 * v_norm / g;
                dz[1] = vy - lugre.z[ix][iy][1] * p.s0 * v_norm / g;

            }

            if (p.stability == true){
                delta_z[0] = (z_ss[0] - lugre.z[ix][iy][0]) / p.dt;
                delta_z[1] = (z_ss[1] - lugre.z[ix][iy][1]) / p.dt;
                dz[0] = std::min(abs(dz[0]), abs(delta_z[0]))*((dz[0] > 0) - (dz[0] < 0));
                dz[1] = std::min(abs(dz[1]), abs(delta_z[1]))*((dz[1] > 0) - (dz[1] < 0));
            }

            lugre.z[ix][iy][0] += dz[0] * p.dt;
            lugre.z[ix][iy][1] += dz[1] * p.dt;

            lugre.f[ix][iy][0] = (p.s0 * lugre.z[ix][iy][0] + p.s1 * dz[0] + p.s2 * vx) * shape_info_var.f_n_grid[ix][iy];
            lugre.f[ix][iy][1] = (p.s0 * lugre.z[ix][iy][1] + p.s1 * dz[1] + p.s2 * vy) * shape_info_var.f_n_grid[ix][iy];

        }
    }
    

}

std::vector<double> FullFrictionModel::approximate_integral(){
    double fx = 0;
    double fy = 0;
    double tau = 0;
    std::vector<double> pos = {0.0, 0.0, 0.0};
    std::vector<double> force = {0.0, 0.0, 0.0};
    std::vector<double> tau_vec = {0.0, 0.0, 0.0};
    
    for (int ix = 0; ix<properties.grid_shape[0]; ix++){
        for (int iy = 0; iy<properties.grid_shape[1]; iy++){
            fx += lugre.f[ix][iy][0];
            fy += lugre.f[ix][iy][1];
            force[0] = lugre.f[ix][iy][0];
            force[1] = lugre.f[ix][iy][1];
            
            pos[0] = position_vec_x[ix];
            pos[1] = position_vec_y[iy];

            tau_vec = utils::crossProduct(pos, force);
            tau += tau_vec[2];
        }
    }
    
    std::vector<double> force_vec = {-fx, -fy, -tau};
    return force_vec;
}

std::vector<double> FullFrictionModel::move_force_to_cop(std::vector<double> force_at_center){
    std::vector<double> f_t(3, 0.0);
    std::vector<double> cop3(3, 0.0);
    std::vector<double> m;
    std::vector<double> force_at_cop(3);

    f_t[0] = force_at_center[0]; f_t[1] = force_at_center[1];
    cop3[0] = shape_info_var.cop[0]; cop3[1] = shape_info_var.cop[1];
    m = utils::crossProduct(cop3, f_t);
    force_at_cop[0] = force_at_center[0];
    force_at_cop[1] = force_at_center[1];
    force_at_cop[2] = force_at_center[2] - m[2];
     
    return force_at_cop;
}


namespace py = pybind11;

//FullFrictionModel hh;
//int ss = hh.step(0.1);

PYBIND11_MODULE(FrictionModelCPPClass, var) {
    var.doc() = "pybind11 example module for a class";
    
    py::class_<FullFrictionModel>(var, "FullFrictionModel")
        .def(py::init<>())
        .def("init", &FullFrictionModel::init)
        .def("step", &FullFrictionModel::step)
        .def("get_force_at_cop", &FullFrictionModel::get_force_at_cop);
}