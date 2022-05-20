import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imageio


def create_output(mesh, solver, output_folder, name):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    results = {}
    for i, n in enumerate(mesh.nodes):
        i1 = 0 + (i - 0) * 6

        results[f"node {int(n[0])}"] = {"Coordinates": n[1:].tolist(),
                                        "Displacement": solver.u[:, i1 : i1 + 3],
                                        "Velocity": solver.v[:, i1 : i1 + 3],
                                        "Acceleration": solver.a[:, i1 : i1 + 3],
                                        "Time": solver.time,
                                        }

    with open(os.path.join(output_folder, name), "wb") as fo:
        pickle.dump(results, fo)
    return results


def plot_geometry(mesh, idx_load, soils, output_folder, name):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # find first soil index
    idx_soil = np.where(np.array(soils))[0][0]

    # plot
    fig, ax = plt.subplots()
    # plot mesh
    ax.plot(np.array(mesh.nodes)[:, 1], np.array(mesh.nodes)[:, 2], marker='o', markersize=1, color="k")
    # plot soil
    ax.plot(np.array(mesh.nodes)[idx_soil:, 1], np.array(mesh.nodes)[idx_soil:, 2], marker='o', markersize=1, color="g")
    # plot load
    ax.plot(np.array(mesh.nodes)[idx_load, 1], np.array(mesh.nodes)[idx_load, 2], marker='D', markersize=6, color="b")

    ax.set_xlabel("X dimension [m]")
    ax.set_ylabel("Y dimension [m]")
    ax.grid()
    plt.savefig(os.path.join(output_folder, f"{name}.png"))
    plt.close()

    # dump mesh file
    with open(os.path.join(output_folder, f"{name}.csv"), "w") as fo:
        fo.write("Node;X coord;Y coord;Z coord\n")
        for m in mesh.nodes:
            fo.write(";".join(map(str, m)) + "\n")

    return


def plot_time_history_node(results, node, output_folder, name, key="Displacement", idx=1):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    fig, ax = plt.subplots()
    ax.plot(results[node]["Time"], results[node][key][:, idx])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(key)
    ax.grid()
    plt.savefig(os.path.join(output_folder, name))
    plt.close()

    return


def make_movie(results, output_folder, name, scale_fct=1000, tmp_folder="tmp", step=10,
               node_start=None, node_end=None, ylabel="Y distance [m]", xlabel="X distance [m]", normalise=False):

    if not os.path.isdir(os.path.join(output_folder, tmp_folder)):
        os.makedirs(os.path.join(output_folder, tmp_folder))

    time = results["node 1"]["Time"]

    list_nodes = list(results.keys())
    if node_start is not None:
        nods = [i.split(" ")[-1] for i in list(list_nodes)]
        idx = nods.index(str(int(node_start)))
        list_nodes = [list_nodes[i] for i in range(idx, len(list_nodes))]

    if node_end is not None:
        nods = [i.split(" ")[-1] for i in list(list_nodes)]
        idx = nods.index(str(int(node_end))) + 1
        list_nodes = [list_nodes[i] for i in range(idx)]

    # initial geometry
    x, y = [], []
    for node in list_nodes:
        x.append(results[node]["Coordinates"][0])
        y.append(results[node]["Coordinates"][1])

    # if normalise: scale_fct = 1 and the y axis displays displacement
    if normalise:
        scale_fct = 1
        x_lim, y_lim = [], []
        for i in range(0, len(time), step):
            for node in list_nodes:
                x_lim.append(results[node]["Coordinates"][0] + results[node]["Displacement"][i, 0] * scale_fct)
                y_lim.append(results[node]["Coordinates"][1] + results[node]["Displacement"][i, 1] * scale_fct)
        y_lim = np.max(np.abs(np.array(y_lim)))


    # plot time steps
    for i in range(0, len(time), step):
        x_t, y_t = [], []
        for node in list_nodes:
            x_t.append(results[node]["Coordinates"][0] + results[node]["Displacement"][i, 0] * scale_fct)
            y_t.append(results[node]["Coordinates"][1] + results[node]["Displacement"][i, 1] * scale_fct)

        fig, ax = plt.subplots()
        ax.plot(x, y, color='k')
        if normalise:
            xlimits = (np.min(x_lim), np.max(x_lim))
            ylimits = (-y_lim, y_lim)
        else:
            xlimits = ax.get_xlim()
            ylimits = ax.get_ylim()
        ax.plot(x_t, y_t, color='b')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Time: {time[i]: .4f} s")
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits)
        ax.grid()
        plt.savefig(os.path.join(output_folder, tmp_folder, f"{str(i).zfill(len(str(len(time))))}.png"))
        plt.close()

    # make de video
    filenames = os.listdir(os.path.join(output_folder, tmp_folder))
    with imageio.get_writer(os.path.join(output_folder, name), mode='I', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(os.path.join(os.path.join(output_folder, tmp_folder), filename))
            writer.append_data(image)
    writer.close()

    shutil.rmtree(os.path.join(output_folder, tmp_folder))


def write_results_csv(results, node_start, node_end, output_folder, name, number_of_periods=2, frequency=20):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # time window
    time = results["node 1"]["Time"]
    t_ini = time[-1] - number_of_periods / frequency
    idx_time = np.where(time > t_ini)[0]

    # collect nodes
    list_nodes = list(results.keys())

    nods = [i.split(" ")[-1] for i in list(list_nodes)]
    idx_ini = nods.index(str(int(node_start)))
    idx_end = nods.index(str(int(node_end))) + 1
    list_nodes = [list_nodes[i] for i in range(idx_ini, idx_end)]

    # collect max/min in time window for each node and write
    with open(os.path.join(output_folder, f"{name}.csv"), "w") as fo:
        fo.write("Node;X coord;Y coord;Z coord;Max Displacement [m];Min Displacement [m]\n")
        # write results for the nodes
        for node in list_nodes:
            fo.write(f"{node};"
                     f"{results[node]['Coordinates'][0]};"
                     f"{results[node]['Coordinates'][1]};"
                     f"{results[node]['Coordinates'][2]};"
                     f"{np.max(results[node]['Displacement'][idx_time, 1])};"
                     f"{np.min(results[node]['Displacement'][idx_time, 1])}\n"
                     )
