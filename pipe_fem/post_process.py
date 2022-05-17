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


def plot_geometry(mesh, load, soils, output_folder, name):

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # find first soil index
    idx_soil = np.where(np.linalg.norm(mesh.nodes[:, 1:] - mesh.nodes[0, 1:], axis=1) -
                        np.linalg.norm(soils - mesh.nodes[0, 1:]) >= 0)[0][0]

    # find first load index
    idx_load = np.where(np.linalg.norm(mesh.nodes[:, 1:] - mesh.nodes[0, 1:], axis=1) -
                        np.linalg.norm(load - mesh.nodes[0, 1:]) >= 0)[0][0]

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
    plt.savefig(os.path.join(output_folder, name))
    plt.close()

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


def make_movie(results, output_folder, name, scale_fct=1000, tmp_folder="tmp", step=10):

    if not os.path.isdir(os.path.join(output_folder, tmp_folder)):
        os.makedirs(os.path.join(output_folder, tmp_folder))

    time = results["node 1"]["Time"]

    # initial geometry
    x, y = [], []
    for node in results.keys():
        x.append(results[node]["Coordinates"][0])
        y.append(results[node]["Coordinates"][1])

    for i in range(0, len(time), step):

        x_t, y_t = [], []
        for node in results.keys():
            x_t.append(results[node]["Coordinates"][0] + results[node]["Displacement"][i, 0] * scale_fct)
            y_t.append(results[node]["Coordinates"][1] + results[node]["Displacement"][i, 1] * scale_fct)

        fig, ax = plt.subplots()
        ax.plot(x, y, color='k')
        xlimits = ax.get_xlim()
        ylimits = ax.get_ylim()
        ax.plot(x_t, y_t, color='b')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Displacement [m]")
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
