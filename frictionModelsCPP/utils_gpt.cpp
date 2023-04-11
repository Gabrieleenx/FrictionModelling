#include <iostream>
#include <cmath>
#include <array>
#include <unordered_map>
#include <vector>

using namespace std;

array<double, 3> cross_product(array<double, 3> a, array<double, 3> b) {
    array<double, 3> result = {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    };
    return result;
}

unordered_map<string, double> vel_to_cop(array<double, 2> cop, unordered_map<string, double> vel_vec) {
    array<double, 3> u = {0, 0, 1};
    array<double, 3> w = {vel_vec["x"], vel_vec["y"], vel_vec["tau"]};
    w = {w[0], w[1], w[2] * u[2]};
    array<double, 3> pos_vex = {cop[0], cop[1], 0};
    array<double, 3> v_tau = cross_product(w, pos_vex);
    double v_x = vel_vec["x"] + v_tau[0];
    double v_y = vel_vec["y"] + v_tau[1];
    double v_tau_new = vel_vec["tau"];
    unordered_map<string, double> result = {
        {"x", v_x},
        {"y", v_y},
        {"tau", v_tau_new}
    };
    return result;
}

std::array<double, 3> update_viscus_scale(FullModel full_model, double gamma, std::array<double, 2> cop) {
    double mu_c = full_model.p["mu_c"];
    double mu_s = full_model.p["mu_s"];
    double s2 = full_model.p["s2"];

    full_model.update_properties(0, 0, 1);
    std::array<double, 3> vel_vec_ = {0, 0, 1};
    std::array<double, 3> vel_vec = vel_to_cop({-cop[0], -cop[1]}, vel_vec_);
    double m = full_model.fn * pow(gamma, 2);
    std::array<double, 3> f = full_model.step(vel_vec);
    double s = abs(f[2] / m);

    // reset parameter back
    full_model.update_properties(mu_c, mu_s, s2);
    return {1, 1, s};
}

double update_radius(FullModel& full_model) {
    auto cop = full_model.cop;
    auto fn = full_model.fn;
    auto mu = full_model.p["mu_c"];
    auto vel_vec_ = std::unordered_map<std::string, double>{{"x", 0}, {"y", 0}, {"tau", 1}};
    auto vel_vec = vel_to_cop(-cop, vel_vec_);
    auto f = full_model.step(vel_vec);
    f = full_model.force_at_cop;
    double gamma;
    if (fn != 0) {
        gamma = std::abs(f["tau"]) / (mu * fn);
    } else {
        gamma = 0;
    }
    return gamma;
}


double elasto_plastic_alpha(double* z, double* z_ss, double z_ba_r, double* v, int dim) {
    double z_norm = 0;
    double z_max = 0;
    double z_ba = 0;
    double alpha = 0;
    double v_norm = 0;
    double eps = 0;
    double c = 0;
    double z_unit[3];
    double v_unit[3];

    // Calculate z_norm and z_max
    for (int i = 0; i < dim; i++) {
        z_norm += z[i] * z[i];
        z_max += z_ss[i] * z_ss[i];
    }
    z_norm = sqrt(z_norm);
    z_max = sqrt(z_max);

    // Calculate z_ba
    z_ba = z_ba_r * z_max;

    // Calculate alpha
    if (z_norm <= z_ba) {
        alpha = 0;
    } else if (z_norm <= z_max) {
        double arg = M_PI * ((z_norm - (z_max + z_ba) / 2) / (z_max - z_ba));
        alpha = 0.5 * sin(arg) + 0.5;
    } else {
        alpha = 1;
    }

    // Calculate eps
    v_norm = 0;
    for (int i = 0; i < dim; i++) {
        v_norm += v[i] * v[i];
        z_unit[i] = z[i] / z_norm;
        v_unit[i] = v[i] / v_norm;
    }
    v_norm = sqrt(v_norm);
    c = 0;
    for (int i = 0; i < dim; i++) {
        c += v_unit[i] * z_unit[i];
    }
    eps = (c + 1) / 2;

    // Update alpha
    alpha *= eps;

    return alpha;
}


std::unordered_map<std::string, double> normalize_force(std::unordered_map<std::string, double> f, double f_t_max, double f_tau_max) {
    std::unordered_map<std::string, double> f_norm;
    f_norm["x"] = f["x"] / f_t_max;
    f_norm["y"] = f["y"] / f_t_max;
    f_norm["tau"] = f["tau"] / f_tau_max;
    return f_norm;
}

int main() {
    array<double, 2> cop = {1, 2};
    unordered_map<string, double> vel_vec = {
        {"x", 3},
        {"y", 4},
        {"tau", 5}
    };
    unordered_map<string, double> result = vel_to_cop(cop, vel_vec);
    cout << "x: " << result["x"] << endl;
    cout << "y: " << result["y"] << endl;
    cout << "tau: " << result["tau"] << endl;
    return 0;
}