#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

void utils::P_x_y::init(std::string shape_name_, double grid_size_, std::vector<int> grid_shape_, double fn){
    grid_size = grid_size_;
    grid_shape = grid_shape_;
    shape_info_var.fn = fn;

    set_shape(shape_name_);
}


void utils::P_x_y::update_shape_info(){
    // check is f_n_grid already matches the correct shape. 
    uint n_row = grid_shape[0];
    uint n_col = grid_shape[1];

    std::vector<std::vector<double>> new_grid(grid_shape[0], std::vector<double>(grid_shape[1], 0.0));

    double area = std::pow(grid_size, 2);
    double fn_new = 0;
    for (int i = 0; i < new_grid.size(); i++) {
        for (int j = 0; j < new_grid[i].size(); j++) {
            new_grid[i][j] = area * utils::shape_function(shape_name, i, j, grid_shape);
            fn_new += new_grid[i][j];
        }
    }

    utils::multiply(new_grid, 1/fn_new);

    shape_info_var.f_n_grid_norm.assign(new_grid.begin(), new_grid.end());

    double cop_x = 0;
    double cop_y = 0;
    double x_pos;
    double y_pos;

    for (int i = 0; i < new_grid.size(); i++) {
        for (int j = 0; j < new_grid[i].size(); j++) {
            x_pos = (i + 0.5 - n_row/2.0)*grid_size;
            x_pos = (j + 0.5 - n_col/2.0)*grid_size;
            cop_x += x_pos * new_grid[i][j];
            cop_y += y_pos * new_grid[i][j];
        }
    }

    shape_info_var.cop_norm.insert(shape_info_var.cop_norm.begin(), {cop_x, cop_y});
}

void utils::P_x_y::set_shape(std::string shape_name_){
    shape_name = shape_name_;
    update_shape_info();
}

void utils::P_x_y::set_fn(double fn){
    shape_info_var.fn = fn;
}

std::string utils::P_x_y::get_shape(){
    return shape_name;
}

utils::shape_info utils::P_x_y::get(double size_){
    shape_info_var.f_n_grid = shape_info_var.f_n_grid_norm;

    utils::multiply(shape_info_var.f_n_grid, shape_info_var.fn);
    shape_info_var.cop = shape_info_var.cop_norm;
    utils::multiply(shape_info_var.cop, size_/grid_size);
    return shape_info_var;
}

double utils::shape_function(std::string shape_name, int ix, int iy, std::vector<int> grid_shape){
    double preasure = 0;
    if (shape_name == "Square"){
        preasure = 0.01;
    }
    return preasure;
}


template <typename T>
void utils::multiply(std::vector<T>& vec, double scalar) {
    for (auto& element : vec) {
            utils::multiply(element, scalar);  // Recurse for nested vectors
    }
}

void utils::multiply(double& element, double scalar){
    element *= scalar;
    }


std::vector<double> utils::crossProduct(const std::vector<double>& v1, const std::vector<double>& v2) {
    std::vector<double> result(3, 0.0);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}


double utils::elasto_plastic(std::vector<double> z, std::vector<double> z_ss, double z_ba_r, std::vector<double> v, int size){
    double z_norm = 0;
    double z_max = 0;
    double v_norm = 0;
    double alpha = 0;

    for (int i = 0; i<size; i++){
        z_norm += pow(z[i], 2);
        z_max += pow(z_ss[i], 2);
        v_norm += pow(v[i], 2);
    }

    z_norm = sqrt(z_norm);
    z_max = sqrt(z_max);
    v_norm = sqrt(v_norm);
    double z_ba = z_ba_r*z_max;

    if (z_norm <= z_ba){
        alpha = 0;
    }else if(z_norm <= z_max){
        alpha = 0.5 * sin(M_PI * ((z_norm - (z_max+z_ba)/2)/(z_max - z_ba))) + 0.5;
    }else{
        alpha = 1;
    }

    if (v_norm != 0 && z_norm != 0){
        double c = 0;
        for (int i = 0; i<size; i++){

            c += v[i]*z[i]/(v_norm * z_norm);
        }
        alpha = ((c + 1)/2)*alpha;
    }

    return alpha;
}


utils::vec utils::vel_to_point(std::vector<double> cop, vec vel){
    std::vector<double> w3(3, 0.0);
    std::vector<double> cop3(3, 0.0);
    std::vector<double> v_tau;
    utils::vec vel_cop;

    w3[2] = vel.tau;
    cop3[0] = cop[0]; cop3[1] = cop[1];
    v_tau = utils::crossProduct(w3, cop3);
    vel_cop.x = vel.x + v_tau[0];
    vel_cop.y = vel.y + v_tau[1];
    vel_cop.tau = vel.tau;
    return vel_cop;
}


std::vector<double> utils::negate_vector(const std::vector<double>& vec){
    std::vector<double> neg_vec(vec.size());
    for (std::size_t i = 0; i < vec.size(); ++i) {
        neg_vec[i] = -vec[i];
    }
    return neg_vec;
}

