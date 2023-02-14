import numpy as np
from scipy.interpolate import interp1d
from pipe_fem.post_process import create_output, plot_geometry


def parameter_update(values, x_data, y_data):
    r"""
    Interpolation of the data.
    If the data is extrapolated the first and last values are used.

    :param values: values to be interpolated
    :param x_data: x data
    :param y_data: y data

    :return: interpolated/extrapolated value
    """
    # interpolate the data
    f = interp1d(x_data, y_data, kind="linear", bounds_error=False, fill_value=(y_data[0], y_data[-1]))
    return f(values)


def collect_peaks(nodes, results_solver, frequency, number_of_periods, key, idx):
    r"""
    Collect the peaks of the displacement of the nodes.

    :param nodes: nodes of the mesh
    :param results_solver: results of the solver
    :param frequency: frequency of the force
    :param number_of_periods: number of periods to be considered
    :param key: key of the results to be considered
    :param idx: index of the results to be considered

    :return: dictionary with the results
    """
    results = {}
    for i, n in enumerate(nodes):
        i1 = 0 + (i - 0) * 6

        results[f"node {int(n[0])}"] = {"Coordinates": n[1:].tolist(),
                                        "Displacement": results_solver.u[:, i1: i1 + 3],
                                        "Velocity": results_solver.v[:, i1: i1 + 3],
                                        "Acceleration": results_solver.a[:, i1: i1 + 3],
                                        "Time": results_solver.time,
                                        }



    # time window
    time = results["node 1"]["Time"]
    t_ini = time[-1] - number_of_periods / frequency
    idx_time = np.where(time > t_ini)[0]

    mean_displacement = [np.mean([np.max(results[node][key][idx_time, idx]), np.abs(np.min(results[node][key][idx_time, idx]))]) for node in results]

    return np.array(mean_displacement)


def write_output(mesh, solver, id_node_force, output_folder, name):
    r"""
    Write the output of the simulation.

    :param mesh: mesh of the pipe
    :param solver: solver of the pipe
    :param id_node_force: index of the node where the load is applied
    :param output_folder: location output folder
    :param name: name of the output file
    """
    print("Saving output")
    # parse data
    create_output(mesh, solver, output_folder, name)
    plot_geometry(mesh, id_node_force, mesh.soil_props, output_folder, "mesh")
