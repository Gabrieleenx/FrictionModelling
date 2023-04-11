#pragma once
#include <string>
#include <vector>


namespace utils {

    struct vec {
        double x;
        double y;
        double tau;
    };

    struct properties
    {            
        std::vector<int> grid_shape; // [x, y]
        double grid_size; // unit [m]
        double mu_c;
        double mu_s;
        double v_s;
        double alpha;
        double s0;
        double s1;
        double s2;
        double dt;
        double z_ba_ratio; // break away from ratio from z_max
        bool stability;
        bool elasto_plastic;
        bool steady_state;
    };

    struct lugre
    {
        std::vector<std::vector<std::vector<double>>> z; // [shape x, shape y, 2]
        std::vector<std::vector<std::vector<double>>> f; // [shape x, shape y, 2]
    };
    
    

    struct shape_info {
        double fn;
        std::vector<double> cop; // [x,y]
        std::vector<double> cop_norm; // [x,y]
        std::vector<std::vector<double>> f_n_grid; // row = ix and col = iy, i.e. f_n_grid[ix, iy]
        std::vector<std::vector<double>> f_n_grid_norm; // row = ix and col = iy, i.e. f_n_grid[ix, iy]
    };

    class P_x_y{
        private:
            std::string shape_name;
            double grid_size; // unit [m]
            std::vector<int> grid_shape; // [x, y]
            shape_info shape_info_var;
            void update_shape_info();
        public:
            P_x_y(){};
            void init(std::string shape_name_, double grid_size_, std::vector<int> grid_shape_, double fn);
            void set_shape(std::string shape_name_);
            void set_fn(double fn);
            std::string get_shape();
            shape_info get(double size_);
    };


    double shape_function(std::string shape_name, int ix, int iy, std::vector<int> grid_shape);
    
    template <typename T>
    void multiply(std::vector<T>& vec, double scalar);

    void multiply(double& element, double scalar);

    std::vector<double> crossProduct(const std::vector<double>& v1, const std::vector<double>& v2);

    double elasto_plastic(std::vector<double> z, std::vector<double> z_ss, double z_ba_r, std::vector<double> v, int size);
}
