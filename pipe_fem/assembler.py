# import packages
import numpy as np
from scipy.sparse import lil_matrix


def gen_stiff(nodes: np.ndarray, elements: np.ndarray, materials: dict):
    r"""
    Global stiffness generation.

    Generates and assembles the global stiffness matrix for the structure.

    :param nodes: Nodes
    :type nodes: np.ndarray
    :param elements: Elements
    :type elements: np.ndarray
    :param materials: Materials properties
    :type elements: dict

    :return k_global: Global stiffness matrix.
    """

    # generation of variable
    k_global = lil_matrix((nodes.shape[0] * 6, nodes.shape[0] * 6))

    # assemblage of stiffness matrix
    for a in range(elements.shape[0]):
        # search for the node
        id_node1 = np.where(elements[a, 1] == nodes[:, 0])[0][0]
        id_node2 = np.where(elements[a, 2] == nodes[:, 0])[0][0]
        # mat = elem[a, 1]

        # compute element size
        elem_size = np.sqrt((nodes[id_node1, 1] - nodes[id_node2, 1])**2 +
                            (nodes[id_node1, 2] - nodes[id_node2, 2])**2 +
                            (nodes[id_node1, 3] - nodes[id_node2, 3])**2)

        # compute rotations
        tx = (nodes[id_node2, 1] - nodes[id_node1, 1]) / elem_size
        ty = (nodes[id_node2, 2] - nodes[id_node1, 2]) / elem_size
        tz = (nodes[id_node2, 3] - nodes[id_node1, 3]) / elem_size

        # read material properties
        E = materials['E']
        Iy = materials['Iy']
        Iz = materials['Iz']
        J = materials['J']
        A = materials['Area']
        poisson = materials["Poisson"]
        G = E / (2 / (1 + poisson))

        # local stiffness matrix
        k_local = stiff_matrix(E, Iy, Iz, G, J, A, elem_size)

        # rotation matrix
        rot = rot_matrix(tx, ty, tz)

        # matrix assembly
        i1 = 0 + (id_node1 - 0) * 6
        i2 = 6 + (id_node1 - 0) * 6
        i3 = 0 + (id_node2 - 0) * 6
        i4 = 6 + (id_node2 - 0) * 6
        elem_dof = np.append(np.arange(i1, i2), np.arange(i3, i4))
        aux = np.dot(np.dot(np.transpose(rot), k_local), rot)

        k_global[elem_dof.reshape(12, 1), elem_dof] = k_global[elem_dof.reshape(12, 1), elem_dof] + aux

    return k_global


def gen_stiff_soil(nodes: np.ndarray, elements: np.ndarray, soil_index: list, soils: dict):
    r"""
    Global stiffness generation.

    Generates and assembles the global stiffness matrix for the structure.

    :param nodes: Nodes
    :type nodes: np.ndarray
    :param elements: Elements
    :type elements: np.ndarray
    :param soil_index: Index where soil exists
    :type soil_index: list
    :param soils: Materials properties
    :type elements: dict

    :return k_global: Global stiffness matrix.
    """

    # generation of variable
    k_soil = lil_matrix((nodes.shape[0] * 6, nodes.shape[0] * 6))

    # find first soil index
    idx_start = np.where(np.array(soil_index))[0][0]

    # assemblage of stiffness matrix
    for a in range(elements.shape[0]):
        # search for the node
        id_node1 = np.where(elements[a, 1] == nodes[:, 0])[0][0]
        id_node2 = np.where(elements[a, 2] == nodes[:, 0])[0][0]

        if any([id_node1, id_node2] < idx_start):
            continue

        # compute element size
        elem_size = np.sqrt((nodes[id_node1, 1] - nodes[id_node2, 1]) ** 2 +
                            (nodes[id_node1, 2] - nodes[id_node2, 2]) ** 2 +
                            (nodes[id_node1, 3] - nodes[id_node2, 3]) ** 2)

        # compute rotations
        tx = (nodes[id_node2, 1] - nodes[id_node1, 1]) / elem_size
        ty = (nodes[id_node2, 2] - nodes[id_node1, 2]) / elem_size
        tz = (nodes[id_node2, 3] - nodes[id_node1, 3]) / elem_size

        if isinstance(soils["Stiffness"][0], np.ndarray):
            kx = np.mean([soils["Stiffness"][0][id_node1], soils["Stiffness"][0][id_node1]])
        else:
            kx = soils["Stiffness"][0]
        if isinstance(soils["Stiffness"][1], np.ndarray):
            ky = np.mean([soils["Stiffness"][1][id_node1], soils["Stiffness"][1][id_node1]])
        else:
            ky = soils["Stiffness"][1]
        if isinstance(soils["Stiffness"][2], np.ndarray):
            kz = np.mean([soils["Stiffness"][2][id_node1], soils["Stiffness"][2][id_node1]])
        else:
            kz = soils["Stiffness"][2]

        kyz = 0

        # local stiffness matrix
        k_local = stiff_soil(kx, ky, kz, kyz, elem_size)

        # rotation matrix
        rot = rot_matrix(tx, ty, tz)

        # matrix assembly
        i1 = 0 + (id_node1 - 0) * 6
        i2 = 6 + (id_node1 - 0) * 6
        i3 = 0 + (id_node2 - 0) * 6
        i4 = 6 + (id_node2 - 0) * 6
        elem_dof = np.append(np.arange(i1, i2), np.arange(i3, i4))
        aux = np.dot(np.dot(np.transpose(rot), k_local), rot)

        k_soil[elem_dof.reshape(12, 1), elem_dof] = k_soil[elem_dof.reshape(12, 1), elem_dof] + aux

    return k_soil


def gen_mass(nodes: np.ndarray, elements: np.ndarray, materials: dict):
    r"""
    Global mass matrix generation.

    Generates and assembles the global mass matrix for the structure.

    :param nodes: Nodes
    :type nodes: np.ndarray
    :param elements: Elements
    :type elements: np.ndarray
    :param materials: Materials properties
    :type elements: dict

    :return m_global: Global mass matrix.
    """

    # generation of variable
    m_global = lil_matrix((nodes.shape[0] * 6, nodes.shape[0] * 6))

    # assemblage of mass matrix
    for a in range(elements.shape[0]):
        # search for the node
        id_node1 = np.where(elements[a, 1] == nodes[:, 0])[0][0]
        id_node2 = np.where(elements[a, 2] == nodes[:, 0])[0][0]
        # mat = elem[a, 1]

        # compute element size
        elem_size = np.sqrt((nodes[id_node1, 1] - nodes[id_node2, 1])**2 +
                            (nodes[id_node1, 2] - nodes[id_node2, 2])**2 +
                            (nodes[id_node1, 3] - nodes[id_node2, 3])**2)

        # compute rotations
        tx = (nodes[id_node2, 1] - nodes[id_node1, 1]) / elem_size
        ty = (nodes[id_node2, 2] - nodes[id_node1, 2]) / elem_size
        tz = (nodes[id_node2, 3] - nodes[id_node1, 3]) / elem_size

        # read material properties
        if isinstance(materials['Density'], np.ndarray):
            # for the case of equivalent linear
            rho = np.mean([materials['Density'][id_node1], materials['Density'][id_node2]])
        else:
            rho = materials['Density']
        A = materials['Area']
        rg = materials['rg']

        # local mass matrix
        m_local = mass_matrix(rho, A, rg, elem_size)
        # rotation matrix
        rot = rot_matrix(tx, ty, tz)

        # matrix assembly
        i1 = 0 + (id_node1 - 0) * 6
        i2 = 6 + (id_node1 - 0) * 6
        i3 = 0 + (id_node2 - 0) * 6
        i4 = 6 + (id_node2 - 0) * 6
        elem_dof = np.append(np.arange(i1, i2), np.arange(i3, i4))
        aux = np.dot(np.dot(np.transpose(rot), m_local), rot)
        m_global[elem_dof.reshape(12, 1), elem_dof] = m_global[elem_dof.reshape(12, 1), elem_dof] + aux

    return m_global


def gen_damp_soil(nodes: np.ndarray, elements: np.ndarray, soil_index: list, soils: dict):
    r"""
    Global stiffness generation.

    Generates and assembles the global stiffness matrix for the structure.

    :param nodes: Nodes
    :type nodes: np.ndarray
    :param elements: Elements
    :type elements: np.ndarray
    :param soil_index: Index where soil exists
    :type soil_index: list
    :param soils: Materials properties
    :type elements: dict

    :return k_global: Global stiffness matrix.
    """

    # generation of variable
    c_soil = lil_matrix((nodes.shape[0] * 6, nodes.shape[0] * 6))

    # find first soil index
    idx_start = np.where(np.array(soil_index))[0][0]

    # assemblage of damping matrix
    for a in range(elements.shape[0]):
        # search for the node
        id_node1 = np.where(elements[a, 1] == nodes[:, 0])[0][0]
        id_node2 = np.where(elements[a, 2] == nodes[:, 0])[0][0]

        if any([id_node1, id_node2] < idx_start):
            continue

        # compute element size
        elem_size = np.sqrt((nodes[id_node1, 1] - nodes[id_node2, 1]) ** 2 +
                            (nodes[id_node1, 2] - nodes[id_node2, 2]) ** 2 +
                            (nodes[id_node1, 3] - nodes[id_node2, 3]) ** 2)

        # compute rotations
        tx = (nodes[id_node2, 1] - nodes[id_node1, 1]) / elem_size
        ty = (nodes[id_node2, 2] - nodes[id_node1, 2]) / elem_size
        tz = (nodes[id_node2, 3] - nodes[id_node1, 3]) / elem_size

        if isinstance(soils["Damping"][0], np.ndarray):
            cx = np.mean([soils["Damping"][0][id_node1], soils["Damping"][0][id_node1]])
        else:
            cx = soils["Damping"][0]
        if isinstance(soils["Damping"][1], np.ndarray):
            cy = np.mean([soils["Damping"][1][id_node1], soils["Damping"][1][id_node1]])
        else:
            cy = soils["Damping"][1]
        if isinstance(soils["Damping"][2], np.ndarray):
            cz = np.mean([soils["Damping"][2][id_node1], soils["Damping"][2][id_node1]])
        else:
            cz = soils["Damping"][2]

        # local damping matrix - in global directions
        c_local = damp_soil(cx, cy, cz, elem_size)

        # rotation matrix
        rot = rot_matrix(tx, ty, tz)

        # matrix assembly
        i1 = 0 + (id_node1 - 0) * 6
        i2 = 6 + (id_node1 - 0) * 6
        i3 = 0 + (id_node2 - 0) * 6
        i4 = 6 + (id_node2 - 0) * 6
        elem_dof = np.append(np.arange(i1, i2), np.arange(i3, i4))
        aux = np.dot(np.dot(np.transpose(rot), c_local), rot)
        c_soil[elem_dof.reshape(12, 1), elem_dof] = c_soil[elem_dof.reshape(12, 1), elem_dof] + aux

    return c_soil


def stiff_matrix(E, Iy, Iz, G, J, A, dL):
    r"""
    Local stiffness generation for a 3D Euler beam.

    Creates the stiffness matrix for one element. Full formulation of Euler beam.
    Based on [1]_.

    .. rubric:: References
    .. [1] Kassimali, A., *Matrix Analysis of Structures.* Cengage Learning, 2Ed, 2012. pp. 456-464.

    :param E: Young modulus.
    :type E: float
    :param Iy: Inertia along y axis.
    :type Iy: float
    :param Iz: Inertia along z axis.
    :type Iz: float
    :param G: Shear modulus.
    :type G: float
    :param J: Polar moment of inertia.
    :type J: float
    :param A: Area.
    :type A: float
    :param dL: Element size.
    :type dL: np.ndarray

    :return k_local: Local stiffness matrix.
    """

    # local stiffness matrix
    k_local = np.zeros((12, 12))

    k_local[0, 0] = E * A / dL
    k_local[1, 1] = 12 * E * Iz / dL**3
    k_local[2, 2] = 12 * E * Iy / dL**3
    k_local[3, 3] = G * J / dL
    k_local[4, 4] = 4 * E * Iy / dL
    k_local[5, 5] = 4 * E * Iz / dL
    k_local[6, 6] = E * A / dL
    k_local[7, 7] = 12 * E * Iz / dL**3
    k_local[8, 8] = 12 * E * Iy / dL**3
    k_local[9, 9] = G * J / dL
    k_local[10, 10] = 4 * E * Iy / dL
    k_local[11, 11] = 4 * E * Iz / dL

    k_local[6, 0] = - E * A / dL
    k_local[0, 6] = - E * A / dL

    k_local[1, 5] = 6 * E * Iz / dL**2
    k_local[5, 1] = 6 * E * Iz / dL**2
    k_local[1, 7] = -12 * E * Iz / dL**3
    k_local[7, 1] = -12 * E * Iz / dL**3
    k_local[1, 11] = 6 * E * Iz / dL**2
    k_local[11, 1] = 6 * E * Iz / dL**2

    k_local[2, 4] = -6 * E * Iy / dL**2
    k_local[4, 2] = -6 * E * Iy / dL**2
    k_local[2, 8] = -12 * E * Iy / dL**3
    k_local[8, 2] = -12 * E * Iy / dL**3
    k_local[2, 10] = -6 * E * Iy / dL**2
    k_local[10, 2] = -6 * E * Iy / dL**2

    k_local[3, 9] = -G * J / dL
    k_local[9, 3] = -G * J / dL

    k_local[4, 8] = 6 * E * Iy / dL**2
    k_local[8, 4] = 6 * E * Iy / dL**2
    k_local[4, 10] = 2 * E * Iy / dL
    k_local[10, 4] = 2 * E * Iy / dL

    k_local[5, 7] = -6 * E * Iz / dL**2
    k_local[7, 5] = -6 * E * Iz / dL**2
    k_local[5, 11] = 2 * E * Iz / dL
    k_local[11, 5] = 2 * E * Iz / dL

    k_local[7, 11] = -6 * E * Iz / dL**2
    k_local[11, 7] = -6 * E * Iz / dL**2

    k_local[8, 10] = 6 * E * Iy / dL**2
    k_local[10, 8] = 6 * E * Iy / dL**2

    return k_local


def mass_matrix(rho, A, rg, dL):
    r"""
    Local mass generation.

    Creates the mass matrix for one element. Full formulation of Euler beam.

    :param rho: Density.
    :type rho: float
    :param A: Area.
    :type A: float
    :param rg: Radius of gyration.
    :type rg: float
    :param dL: Element size.
    :type dL: float

    :return m_local: Local mass matrix.
    """

    # local mass matrix
    m_local = np.zeros((12, 12))

    m_local[0, 0] = 1/3
    m_local[1, 1] = 13/35
    m_local[2, 2] = 13/35
    m_local[3, 3] = (rg**2)/3
    m_local[4, 4] = (dL**2)/105
    m_local[5, 5] = (dL**2)/105
    m_local[6, 6] = 1/3
    m_local[7, 7] = 13/35
    m_local[8, 8] = 13/35
    m_local[9, 9] = (rg**2)/3
    m_local[10, 10] = (dL**2)/105
    m_local[11, 11] = (dL**2)/105

    m_local[6, 0] = 1/6
    m_local[0, 6] = 1/6

    m_local[1, 5] = 11 * dL / 210
    m_local[5, 1] = 11 * dL / 210
    m_local[1, 7] = 9 / 70
    m_local[7, 1] = 9 / 70
    m_local[1, 11] = -13 * dL / 420
    m_local[11, 1] = -13 * dL / 420

    m_local[2, 4] = -11 * dL / 210
    m_local[4, 2] = -11 * dL / 210
    m_local[2, 8] = 9 / 70
    m_local[8, 2] = 9 / 70
    m_local[2, 10] = 13 * dL / 420
    m_local[10, 2] = 13 * dL / 420

    m_local[3, 9] = (rg**2) / 6
    m_local[9, 3] = (rg**2) / 6

    m_local[4, 8] = -13 * dL / 420
    m_local[8, 4] = -13 * dL / 420
    m_local[4, 10] = -dL**2 / 140
    m_local[10, 4] = -dL**2 / 140

    m_local[5, 7] = 13 * dL / 420
    m_local[7, 5] = 13 * dL / 420
    m_local[5, 11] = -(dL**2) / 140
    m_local[11, 5] = -(dL**2) / 140

    m_local[7, 11] = -11 * dL / 210
    m_local[11, 7] = -11 * dL / 210

    m_local[8, 10] = 11 * dL / 210
    m_local[10, 8] = 11 * dL / 210

    m_local = m_local.dot(rho * A * dL)

    return m_local


def rot_matrix(tx, ty, tz):
    r"""
    Rotation matrix.

    Creates the rotation matrix that transforms the local in the global axis system.
    It follow the approach proposed in [5]_, [6]_:

    :param tx: Young modulus.
    :type tx: float
    :param ty: Inertia along y axis.
    :type ty: float
    :param tz: Inertia along z axis.
    :type tz: float

    :return rot_full: Local stiffness matrix.

    .. rubric:: References
    .. [5] Krishnamoorthy, *Finite Element Analysis. Theory and programming.* McGraw-Hill, 1994. pp. 243-248.
    .. [6] Smith and Griffiths, *Programming the Finite Element Method.* Wiley, 2004. pp. 137-137.
    """

    # rotation matrix
    T_beta = np.zeros((3, 3))
    T_gamma = np.zeros((3, 3))
    T_alpha = np.zeros((3, 3))
    rot_full = np.zeros((12, 12))

    # Finite Element Analysis:
    # Theory and programming
    # Krishnamoorthy
    # pp 243-248
    # also: Smith and Griffiths
    # pp 137-137
    Cx = tx
    Cy = ty
    Cz = tz

    if Cz == 0 and Cx == 0:
        # vertical beam
        alpha = np.pi/2  #np.arcsin(zk / np.sqrt(xk**2+xk**2))
        T_alpha[0, 0] = 0
        T_alpha[0, 1] = Cy
        T_alpha[0, 2] = 0
        T_alpha[1, 0] = -Cy * np.cos(alpha)
        T_alpha[1, 1] = 0
        T_alpha[1, 2] = np.sin(alpha)
        T_alpha[2, 0] = Cy * np.sin(alpha)
        T_alpha[2, 1] = 0
        T_alpha[2, 2] = np.cos(alpha)

        rot = T_alpha
    else:

        T_beta[0, 0] = Cx / np.sqrt(Cx**2 + Cz**2)
        T_beta[0, 1] = 0
        T_beta[0, 2] = Cz / np.sqrt(Cx**2 + Cz**2)
        T_beta[1, 0] = 0
        T_beta[1, 1] = 1
        T_beta[1, 2] = 0
        T_beta[2, 0] = -Cz / np.sqrt(Cx**2 + Cz**2)
        T_beta[2, 1] = 0
        T_beta[2, 2] = Cx / np.sqrt(Cx**2 + Cz**2)

        T_gamma[0, 0] = np.sqrt(Cx**2 + Cz**2)
        T_gamma[0, 1] = Cy
        T_gamma[0, 2] = 0
        T_gamma[1, 0] = -Cy
        T_gamma[1, 1] = np.sqrt(Cx**2 + Cz**2)
        T_gamma[1, 2] = 0
        T_gamma[2, 0] = 0
        T_gamma[2, 1] = 0
        T_gamma[2, 2] = 1

        alpha = np.arctan(Cy / np.sqrt(Cx**2 + Cz**2))
        T_alpha[0, 0] = 1
        T_alpha[0, 1] = 0
        T_alpha[0, 2] = 0
        T_alpha[1, 0] = 0
        T_alpha[1, 1] = np.cos(alpha)
        T_alpha[1, 2] = np.sin(alpha)
        T_alpha[2, 0] = 0
        T_alpha[2, 1] = -np.sin(alpha)
        T_alpha[2, 2] = np.cos(alpha)

        rot = T_alpha.dot(T_gamma).dot(T_beta)

    for i in range(4):
        a = 0 + (i-0) * 3
        rot_full[a:a + 3, a: a + 3] = rot

    return rot_full


def damping(damp):
    r"""
    Generates the Rayleigh damping coefficients

    :param damp: Damping parameters:

                    damp[0] - :math:`f_1` - frequency [Hz]

                    damp[1] - :math:`d_1` - damping for frequency :math:`f_1`

                    damp[2] - :math:`f_2` - frequency [Hz]

                    damp[3] - :math:`d_2` - damping for frequency :math:`f_2`

    :type damp: float

    :return coefs: Damping coefficients:

                    coefs[0] - :math:`\alpha` - mass parameter

                    coefs[1] - :math:`\beta` - stiffness parameter
    """

    # import packages
    f1 = damp[0]
    d1 = damp[1]
    f2 = damp[2]
    d2 = damp[3]

    if f1 == f2:
        raise SystemExit('Frequencies for the Rayleigh damping are the same.')

    # damping matrix
    damp_mat = 1/2 * np.array([[1/(2 * np.pi * f1), 2 * np.pi * f1],
                              [1/(2 * np.pi * f2), 2 * np.pi * f2]])
    damp_qsi = np.array([d1, d2])

    # solution
    coefs = np.linalg.solve(damp_mat, damp_qsi)

    return coefs[0], coefs[1]


def abs_matrix(rho, vp, vs, A, param, dL):
    r"""
    Local absorbing boundary conditions

    :param rho: Density
    :type rho: float
    :param vp: Compression wave velocity
    :type vp: float
    :param vs: Shear wave velocity
    :type vs: float
    :param A: Area
    :type A: float
    :param param: Parameters for the absorbing boundaries
    :type param: list
    :param dL: Element size.
    :type dL: np.ndarray

    :return k_local: Local absorbing matrix.
    """

    # local stiffness matrix
    abs_local = np.zeros((12, 12))

    abs_local[0, 0] = param[0] * rho * vp * 0.5 * dL
    abs_local[1, 1] = param[1] * rho * vs * 0.5 * dL
    abs_local[2, 2] = param[1] * rho * vs * 0.5 * dL

    abs_local[6, 6] = param[0] * rho * vp * 0.5 * dL
    abs_local[7, 7] = param[1] * rho * vs * 0.5 * dL
    abs_local[8, 8] = param[1] * rho * vs * 0.5 * dL

    return abs_local.dot(A)


def external_force(nodes, force_properties):
    """
    Compute external force

    :param nodes: list of nodes
    :param force_properties: Dictionary with force settings
    :return: time, force, index of the node where the load is applied
    """
    time = np.linspace(0,
                       force_properties["Time"],
                       int(np.ceil(force_properties["Time"] / force_properties["Time_step"] + 1)))

    # generation of variable
    force = lil_matrix((nodes.shape[0] * 6, len(time)))

    id_nodes = []
    for id_f, force_coord in enumerate(force_properties["Coordinates"]):

        # find node where load is applied: always the closest to
        id_node = np.argmin(np.sqrt(np.sum((nodes[:, 1:] - force_coord)**2, axis=1)))

        i1 = 0 + (id_node - 0) * 6
        i2 = 6 + (id_node - 0) * 6

        # determine DOF of the load
        aux = np.zeros(6)
        for i, val in enumerate(force_properties["DOF"][id_f]):
            if val == "1":
                aux[i] = 1

        # add force
        force[i1:i2, :] = force[i1:i2, :] + \
            (np.tile(force_properties["Amplitude"][id_f] * \
                np.sin(2 * np.pi * float(force_properties["Frequency"][id_f]) * time + force_properties["Phase"][id_f]), (6, 1)).T * aux).T
        # add id nodes
        id_nodes.append(id_node)

    return time, force, id_nodes


def stiff_soil(kxx, kyy, kzz, kyz, dL):
    r"""
    Local spring stiffness matrix.

    Generates the stiffness matrix for one element.

    :param kxx: Spring stiffness along x direction.
    :type kxx: float
    :param kyy: Spring stiffness along y direction.
    :type kyy: float
    :param kzz: Spring stiffness along z direction.
    :type kzz: float
    :param kyz: Spring stiffness yz direction.
    :type kyz: float
    :param dL: Element size.
    :type dL: float

    :return k_local: Local soil stiffness matrix.
    """


    # local stiffness matrix
    k_local = np.zeros((12, 12))

    k_local[0, 0] = kxx * 0.5 * dL
    k_local[1, 1] = kyy * 0.5 * dL
    k_local[2, 2] = kzz * 0.5 * dL
    k_local[3, 3] = 0 * dL
    k_local[4, 4] = 0 * kyy
    k_local[5, 5] = 0 * kzz
    k_local[6, 6] = kxx * 0.5 * dL
    k_local[7, 7] = kyy * 0.5 * dL
    k_local[8, 8] = kzz * 0.5 * dL
    k_local[9, 9] = 0 * dL
    k_local[10, 10] = 0 * kyy
    k_local[11, 11] = 0 * kzz

    k_local[1, 2] = kyz * 0.5 * dL
    k_local[2, 1] = kyz * 0.5 * dL
    k_local[7, 8] = kyz * 0.5 * dL
    k_local[8, 7] = kyz * 0.5 * dL

    return k_local


def damp_soil(cx, cy, cz, dL):
    r"""
    Local spring damping matrix.

    Generates the damping matrix for one element.

    :param cx: Spring damping along x direction.
    :type cx: float
    :param cy: Spring damping along y direction.
    :type cy: float
    :param cz: Spring damping along z direction.
    :type cz: float
    :param dL: Element size.
    :type dL: float

    :return c_local: Local soil damping matrix.
    """
    # local stiffness matrix
    c_local = np.zeros((12, 12))

    c_local[0, 0] = cx * 0.5 * dL
    c_local[1, 1] = cy * 0.5 * dL
    c_local[2, 2] = cz * 0.5 * dL
    c_local[3, 3] = 0 * cx
    c_local[4, 4] = 0 * cy
    c_local[5, 5] = 0 * cz
    c_local[6, 6] = cx * 0.5 * dL
    c_local[7, 7] = cy * 0.5 * dL
    c_local[8, 8] = cz * 0.5 * dL
    c_local[9, 9] = 0 * cx
    c_local[10, 10] = 0 * cy
    c_local[11, 11] = 0 * cz

    return c_local


def absorbing_bc(nodes: np.ndarray, elements: np.ndarray, materials: dict, param: list = [1, 1]):
    r"""
    Absorbing boundary conditions.

    Compute absorbing boundary force at the extremity nodes.

    :param nodes: Nodes
    :type nodes: np.ndarray
    :param elements: Elements
    :type elements: np.ndarray
    :param materials: Materials properties
    :type elements: dict
    :param param: Parameters for absorbing BC
    :type elements: list

    :return absorbing: Global absorbing matrix.
    """

    # generation of variable
    absorbing = lil_matrix((nodes.shape[0] * 6, nodes.shape[0] * 6))

    nodes_ext = [nodes[0][0], nodes[-1][0]]

    # assemblage of stiffness matrix
    for a in range(elements.shape[0]):

        if any([True for i in elements[a, 1:] if i in nodes_ext]):

            # search for the node
            id_node1 = np.where(elements[a, 1] == nodes[:, 0])[0][0]
            id_node2 = np.where(elements[a, 2] == nodes[:, 0])[0][0]

            # compute element size
            elem_size = np.sqrt((nodes[id_node1, 1] - nodes[id_node2, 1])**2 +
                                (nodes[id_node1, 2] - nodes[id_node2, 2])**2 +
                                (nodes[id_node1, 3] - nodes[id_node2, 3])**2)

            # compute rotations
            tx = (nodes[id_node2, 1] - nodes[id_node1, 1]) / elem_size
            ty = (nodes[id_node2, 2] - nodes[id_node1, 2]) / elem_size
            tz = (nodes[id_node2, 3] - nodes[id_node1, 3]) / elem_size

            # read material properties
            E = materials['E']
            # read material properties
            if isinstance(materials['Density'], np.ndarray):
                # for the case of equivalent linear
                rho = np.mean([materials['Density'][id_node1], materials['Density'][id_node2]])
            else:
                rho = materials['Density']
            area = materials["Area"]
            poisson = materials["Poisson"]
            Ec = E * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson))
            G = E / (2 * (1 + poisson))
            vp = np.sqrt(Ec / rho)
            vs = np.sqrt(G / rho)

            # local stiffness matrix
            abs_local = abs_matrix(rho, vp, vs, area, param, elem_size)
            if elements[a, 1] in nodes_ext:
                abs_local[6:, 6:] = 0
            else:
                abs_local[:6, :6] = 0

            # rotation matrix
            rot = rot_matrix(tx, ty, tz)

            # matrix assembly
            i1 = 0 + (id_node1 - 0) * 6
            i2 = 6 + (id_node1 - 0) * 6
            i3 = 0 + (id_node2 - 0) * 6
            i4 = 6 + (id_node2 - 0) * 6
            elem_dof = np.append(np.arange(i1, i2), np.arange(i3, i4))
            aux = np.dot(np.dot(np.transpose(rot), abs_local), rot)

            absorbing[elem_dof.reshape(12, 1), elem_dof] = absorbing[elem_dof.reshape(12, 1), elem_dof] + aux

    return absorbing
