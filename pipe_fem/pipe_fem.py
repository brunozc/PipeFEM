import itertools
import numpy as np

from pipe_fem.mesher import Generate
from pipe_fem.solver import Solver
from pipe_fem.utils import parameter_update, collect_peaks, write_output
import pipe_fem.assembler as assembler


def pipe_fem(points, element_size, soil_properties, pipe_properties, force, settings, reduction_values=None,
             output_folder="./", name="data.pickle", max_iterations=100, tol=1e-2, nb_cycles=4):
    r"""
    Main program to run the finite element analysis

    points: list of points defining the pipe
    element_size: size of the elements
    soil_properties: dictionary with soil properties
    pipe_properties: dictionary with pipe properties
    force: dictionary with force properties
    settings: dictionary with Rayleigh and Newmark settings
    reduction_values: list of lists with the values to update the parameters. If None, solves linear elastic problem
    output_folder: folder to save the results
    name: name of the file to save the results
    max_iterations: maximum number of iterations for equivalent linear elastic analysis
    tol: tolerance for equivalent linear elastic analysis
    nb_cycles: number of cycles to compute the maximum displacement for equivalent linear elastic analysis
    """

    print("Generating mesh")
    # generate mesh
    mesh = Generate(points[0], points[1], points[2], points[3], points[4], element_size)
    mesh.soil_materials(soil_properties["Y ground"])

    # check if equivalent linear soil is needed
    if reduction_values is not None:
        iter = 1
        converged = False
        initial_displacement = np.zeros(mesh.nodes.shape[0])
        initial_density = np.copy(pipe_properties["Density"])
        initial_stiffness = np.copy(soil_properties["Stiffness"])
        initial_damping = np.copy(soil_properties["Damping"])

        while (iter<max_iterations) and (not converged):
            print(f"Iteration: {iter}")
            # index where the force is applied
            idx_force = [[pos for pos, char in enumerate(f) if char == "1"] for f in force["DOF"]]
            idx_force = list(set(itertools.chain.from_iterable(idx_force)))[0]

            pipe_properties["Density"] = initial_density * parameter_update(initial_displacement, reduction_values[0], reduction_values[1])
            soil_properties["Stiffness"][idx_force] = initial_stiffness[idx_force] * parameter_update(initial_displacement, reduction_values[0], reduction_values[2])
            soil_properties["Damping"][idx_force] = initial_damping[idx_force] * parameter_update(initial_displacement, reduction_values[0], reduction_values[3])
            results, id_node_force = run_solve(mesh, pipe_properties, soil_properties, force, settings)

            displacement = collect_peaks(mesh.nodes, results, min(force["Frequency"]),
                                         nb_cycles, "Displacement", idx_force)
            # check convergency
            error = np.linalg.norm(np.abs((displacement - initial_displacement) / displacement))
            print(f"Iteration {iter} error: {round(error * 100, 1)}%")
            if error < tol:
                converged = True
            else:
                initial_displacement = np.copy(displacement)

            iter += 1  # update iteration counter
    else:
        results, id_node_force = run_solve(mesh, pipe_properties, soil_properties, force, settings)
    # write results
    write_output(mesh, results, id_node_force, output_folder, name)


def run_solve(mesh, pipe_properties, soil_properties, force, settings):
    r"""
    Run the finite element analysis

    mesh: mesh object
    pipe_properties: dictionary with pipe properties
    soil_properties: dictionary with soil properties
    force: dictionary with force properties
    settings: dictionary with solver settings

    return: solver object, index on the node where the force is applied
    """

    # matrix generation and assemblage
    print("Assembling matrices")
    k_pipe = assembler.gen_stiff(mesh.nodes, mesh.elements, pipe_properties)
    m_pipe = assembler.gen_mass(mesh.nodes, mesh.elements, pipe_properties)
    k_soil = assembler.gen_stiff_soil(mesh.nodes, mesh.elements, mesh.soil_props, soil_properties)
    c_soil = assembler.gen_damp_soil(mesh.nodes, mesh.elements, mesh.soil_props, soil_properties)

    # absorbing BC
    f_abs = assembler.absorbing_bc(mesh.nodes, mesh.elements, pipe_properties)

    # Rayleigh damping
    alpha, beta = assembler.damping(settings['Damping_parameters'])
    c_pipe = m_pipe.tocsr().dot(alpha) + k_pipe.tocsr().dot(beta)

    # all matrix
    M = m_pipe
    K = k_pipe + k_soil
    C = c_pipe + c_soil

    # force
    print("Generating external force")
    time, force_ext, id_node_force = assembler.external_force(mesh.nodes, force)

    # solver
    print("Solver started")
    solver = Solver(K.shape[0])
    solver.newmark(settings, M, C, K, force_ext, f_abs, force["Time_step"], time)

    return solver, id_node_force
