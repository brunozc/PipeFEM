import matplotlib.pylab as plt
import pickle


with open("./results/data.pickle", "rb") as fi:
    data = pickle.load(fi)


fig, ax = plt.subplots()
# ax.plot(data["node 1"]["Time"], data["node 1"]["Displacement"][:, 1], label="node 1")
# ax.plot(data["node 20"]["Time"], data["node 20"]["Displacement"][:, 1], label="node 10")
ax.plot(data["node 5000"]["Time"], data["node 5000"]["Displacement"][:, 1], label="node 5000")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Vertical displacement [m]")
ax.grid()
plt.legend()
plt.show()


from pipe_fem.post_process import make_movie
make_movie(data, "res", "Disp.gif", fct=0.01, step=5)
