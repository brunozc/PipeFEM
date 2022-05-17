import numpy as np
import matplotlib.pylab as plt


class Quadratic:
    """
    Quadratic Bezier function
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, element_size: float = 0.5):
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        self.element_size = element_size

        self.length = 0
        self.size = []
        self.nb_divs = int(np.ceil((np.sqrt(np.sum((self.p1 - self.p0) ** 2)) +
                                    np.sqrt(np.sum((self.p2 - self.p1) ** 2))) / element_size))

        self.coordinates = np.zeros((self.nb_divs, len(self.p0)))
        self.compute_coordinates()
        self.compute_length()

    def compute_coordinates(self):
        for i, t in enumerate(np.linspace(0, 1, self.nb_divs)):
            self.coordinates[i, :] = (1 - t) ** 2 * self.p0 +\
                                     2 * t * (1 - self.nb_divs) * self.p1 +\
                                     t ** 2 * self.p2

    def compute_length(self):
        self.size = np.sqrt((self.coordinates- self.coordinates[0])**2)
        self.element_size = np.cumsum(self.size)


if __name__ == "__main__":

    p1 = np.array([-50, 50])
    p2 = np.array([0, 0])
    p3 = np.array([50, 0])

    p = Quadratic(p1, p2, p3)
    plt.plot(p.coordinates[:, 0], p.coordinates[:, 1], color="b")
    plt.plot(p1[0], p1[1], marker="x", color="r")
    plt.plot(p2[0], p2[1], marker="x", color="r")
    plt.plot(p3[0], p3[1], marker="x", color="r")
    plt.show()
