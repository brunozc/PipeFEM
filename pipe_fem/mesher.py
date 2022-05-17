import numpy as np
from pipe_fem.bezier import Quadratic


class Generate:
    def __init__(self, p1, p2, p3, p4, p5, element_size):

        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.p4 = np.array(p4)
        self.p5 = np.array(p5)

        # straight part 1
        nb_ele_1 = int(np.ceil(np.sqrt(np.sum((self.p1 - self.p2) ** 2)) / element_size))
        coords_1 = split(self.p1, self.p2, nb_ele_1)

        # curve part
        curve = Quadratic(self.p2, self.p3, self.p4, element_size=element_size)

        # straight part 2
        nb_ele_2 = int(np.ceil(np.sqrt(np.sum((self.p4 - self.p5) ** 2)) / element_size))
        coords_2 = split(self.p4, self.p5, nb_ele_2)

        # nodes
        aux_nodes = np.vstack([coords_1, curve.coordinates[1:], coords_2[1:]])
        self.nodes = np.insert(aux_nodes, 0, np.linspace(1, len(aux_nodes), len(aux_nodes)), axis=1)

        # elements
        elem = []
        for i in range(len(self.nodes) - 1):
            elem.append([i+1, self.nodes[i, 0], self.nodes[i + 1, 0]])

        self.elements = np.array(elem)

        # soil properties all False
        self.soil_props = [False] * len(self.nodes)

    def soil_materials(self, coord_start):

        idx = np.where(np.linalg.norm(np.array(self.nodes)[:, 1:]-np.array(self.nodes)[0, 1:], axis=1)
                       >= np.linalg.norm(np.array(coord_start)-np.array(self.nodes)[0, 1:]))[0]

        for i in idx:
            self.soil_props[i] = True


def split(start, end, segments):
    delta = (end - start) / segments
    points = []
    for i in range(1, segments):
        points.append(start + i * delta)
    return np.array([start] + points + [end])


if __name__ == "__main__":
    p1 = [-150, -150, 0]
    p2 = [-50, -50, 0]
    p3 = [0, 0, 0]
    p4 = [50, 0, 0]
    p5 = [150, 0, 0]
    mesh = Generate(p1, p2, p3, p4, p5, 25)
