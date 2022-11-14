from pipe_fem.pipe_fem import pipe_fem

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
force = {"Coordinates": [75, 0, 0],
         "Frequency": 20,
         "Amplitude": 100e6,
         "DOF": "010000",
         "Time": 3,
         "Time_step": 0.005}



# # Force settings
# force = {"Coordinates": [[75, 0, 0], [50, 0, 0]]
#          "Frequency": [20, 20],
#          "Amplitude": [100e6, 100e6],
#          "Phase": [[pi, 2 *pi]],
#          "DOF": ["010000", "010000"],
#          "Time": 5,
#          "Time_step": 0.005}

# Rayleigh and Newmark settings
settings = {"Damping_parameters": [1, 0.01, 100, 0.01],
            "Newmark_beta": 0.25,
            "Newmark_gamma": 0.5,
            }

# run code
pipe_fem([p1, p2, p3, p4, p5], element_size, soil_properties, pipe_properties, force, settings,
         output_folder="results", name="data.pickle")
