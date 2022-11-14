import shutil
import pickle
import unittest

import numpy as np

from pipe_fem.pipe_fem import pipe_fem
from pipe_fem.post_process import make_movie, write_results_csv

TOL = 1e-6


class TestPipe(unittest.TestCase):
    def setUp(self):
        # Geometry
        p1 = [-500, 500, 0]
        p2 = [-50, 50, 0]
        p3 = [0, 0, 0]
        p4 = [50, 0, 0]
        p5 = [750, 0, 0]
        element_size = 0.25

        # Soil properties
        soil_properties = {"Y ground": 100,
                        "Stiffness": [70e3, 550e3, 550e3],
                        "Damping": [5e3, 5e3, 5e3],
                        }
        # Pipe properties
        pipe_properties = {"E": 210e9,
                           "Density": 2000,
                           "Poisson": 0.2,
                           "Area": 0.001,
                           "Iy": 0.00035,
                           "Iz": 0.00035,
                           "J": 0.0077,
                           "rg": 0.02,
                           }

        # Force settings
        force = {"Coordinates": [[75, 0, 0]],
                "Frequency": [20],
                "Amplitude": [100e6],
                "Phase": [np.pi],
                "DOF": ["010000"],
                "Time": 3,
                "Time_step": 0.005}

        # Rayleigh and Newmark settings
        settings = {"Damping_parameters": [1, 0.01, 100, 0.01],
                    "Newmark_beta": 0.25,
                    "Newmark_gamma": 0.5,
                    }

        # run code
        pipe_fem([p1, p2, p3, p4, p5], element_size, soil_properties, pipe_properties, force, settings,
                output_folder="./results_test", name="data.pickle")

        with open(r"./results_test/data.pickle", "rb") as fi:
            data = pickle.load(fi)
        # write amplitudes
        write_results_csv(data, 3129, 3150, "results_test", "amplitudes",
                          number_of_periods=2, frequency=20)

    def test_amplitudes(self):
        with open(r"./results_test/amplitudes.csv", "r", encoding="UTF-8") as fi:
            data = fi.read().splitlines()
        data = [dat.split(";") for dat in data]


        with open(r"./tests/data/amplitudes.csv", "r", encoding="UTF-8") as fi:
            test_data = fi.read().splitlines()
        test_data = [dat.split(";") for dat in test_data]

        # compare two lists
        for val1, val2 in zip(data[1:], test_data[1:]):
            assert all(np.array(val1[1:]).astype(float) - np.array(val2[1:]).astype(float) <= TOL)

    def test_pickle(self):
        with open(r"./results_test/data.pickle", "rb") as fi:
            data = pickle.load(fi)

        with open("./tests/data/data.pickle", "rb") as fi:
            test_data = pickle.load(fi)

        # compare two dictionaries
        assert compare_dics(data, test_data)

    def tearDown(self):
        shutil.rmtree("./results_test")


def compare_dics(dic1, dic2):
    result = []
    for key in dic1:
        res = False
        for k in dic1[key].keys():
            if np.all(np.abs(np.array(dic1[key][k]) - np.array(dic2[key][k])) < TOL):
                res = True
            else:
                res = False
            result.append(res)

    if all(result):
        result = True
    else:
        result = False
    return result

if __name__ == "__main__":
    unittest.main()
