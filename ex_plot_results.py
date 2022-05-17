import pickle
import matplotlib.pylab as plt
from pipe_fem.post_process import make_movie

# load results
with open("./results/data.pickle", "rb") as fi:
    data = pickle.load(fi)

# plot time history
fig, ax = plt.subplots()
ax.set_position([0.2, 0.12, 0.7, 0.8])
ax.plot(data["node 1"]["Time"], data["node 1"]["Displacement"][:, 1], label="node 1")
ax.plot(data["node 20"]["Time"], data["node 20"]["Displacement"][:, 1], label="node 20")
ax.plot(data["node 4500"]["Time"], data["node 4500"]["Displacement"][:, 1], label="node 4500")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Vertical displacement [m]")
ax.grid()
plt.legend()
plt.show()

# make movie
make_movie(data, "results", "displacement_field.gif", scale_fct=1e4, step=2)
