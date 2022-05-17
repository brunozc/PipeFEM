from pipe_fem.mesher import Generate
from pipe_fem.solver import Solver
from pipe_fem.post_process import create_output, plot_geometry
import pipe_fem.assembler as assembler


def pipe_fem(points, element_size, soil_properties, pipe_properties, force, settings,
             output_folder="./", name="data.pickle"):

    print("Generating mesh")
    # generate mesh
    mesh = Generate(points[0], points[1], points[2], points[3], points[4], element_size)
    mesh.soil_materials(soil_properties["Coordinates"])

    print("Assembling matrices")
    # matrix generation and assemblage
    k_pipe = assembler.gen_stiff(mesh.nodes, mesh.elements, pipe_properties)
    m_pipe = assembler.gen_mass(mesh.nodes, mesh.elements, pipe_properties)
    k_soil = assembler.gen_stiff_soil(mesh.nodes, mesh.elements, soil_properties)
    c_soil = assembler.gen_damp_soil(mesh.nodes, mesh.elements, soil_properties)

    # absorbing BC
    f_abs = assembler.absorbing_bc(mesh.nodes, mesh.elements, pipe_properties)

    # Rayleigh damping
    alpha, beta = assembler.damping(settings['Damping_parameters'])
    c_pipe = m_pipe.tocsr().dot(alpha) + k_pipe.tocsr().dot(beta)

    # all matrix
    M = m_pipe
    K = k_pipe + k_soil
    C = c_pipe + c_soil

    print("Generating external force")
    # force
    time, force_ext = assembler.external_force(mesh.nodes, force)

    print("Solver")
    # solver
    solver = Solver(K.shape[0])
    solver.newmark(settings, M, C, K, force_ext, f_abs, force["Time_step"], time)

    print("Saving output")
    # parse data
    create_output(mesh, solver, output_folder, name)
    plot_geometry(mesh, force["Coordinates"], soil_properties["Coordinates"], output_folder, "mesh.png")
