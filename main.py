# uncomment the script to run
#script = "tests/2dLuGreDriftTest.py"
#script = "tests/computational_speed_test.py"
#script = "tests/discrete_force_steps.py"
#script = "tests/distributed_vs_reduced.py"
#script = "tests/ellipsoid.py"
#script = "tests/num_cells.py"
#script = "tests/num_cells_reduced_model.py"
#script = "tests/num_Ls_segments_reduced_model.py"
#script = "tests/planar_slip_stick.py"
#script = "tests/planar_slip_stick_diff_CoM.py"
#script = "tests/plot_Limit_surface.py"
#script = "tests/pre_computational_speed_test_LS.py"
#script = "tests/test_size_change.py"

#script = "plotlyVizualization/plotly_viz.py"

## With ode solver
#script = "test_ode_solver/planar_slip_stick_solver.py"
#script = "test_ode_solver/2dLuGreDriftTest_solver.py"
#script = "test_ode_solver/distributed_vs_reduced_solver.py"
#script = "test_ode_solver/distributed_vs_reduced_solver_bounds.py"
#script = "test_ode_solver/planar_slip_stick_change_surf_solver.py"
#script = "test_ode_solver/planar_slip_stick_sim_time.py"
#script = "test_ode_solver/planar_slip_stick_solver_diff_CoM.py"


# Run the script
with open(script) as f:
    exec(f.read())