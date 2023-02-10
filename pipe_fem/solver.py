import numpy as np
from scipy.sparse.linalg import splu, spsolve
from tqdm import tqdm


def init(m_global, c_global, k_global, force_ini, u, v):
    r"""
    Calculation of the initial conditions - acceleration for the first time-step.

    :param m_global: Global mass matrix
    :param c_global: Global damping matrix
    :param k_global: Global stiffness matrix
    :param force_ini: Initial force
    :param u: Initial conditions - displacement
    :param v: Initial conditions - velocity

    :return a: Initial acceleration
    """

    k_part = k_global.dot(u)
    c_part = c_global.dot(v)

    return spsolve(m_global.tocsr(), force_ini - c_part - k_part)


class Solver:
    def __init__(self, number_equations):
        # define initial conditions
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        # define variables
        self.u = []
        self.v = []
        self.a = []
        self.time = []

        return

    def newmark(self, settings, M, C, K, F, F_abs, t_step, time):
        """
        Newmark integration scheme.
        Incremental formulation.

        :param settings: dictionary with the integration settings
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param F_abs: Absorbing boundary force matrix
        :param t_step: time step for the analysis
        :param time: time array for the analysis
        :return:
        """

        # constants for the Newmark integration
        beta = settings["Newmark_beta"]
        gamma = settings["Newmark_gamma"]

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, 0].toarray()
        d_force = d_force[:, 0]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        dv = np.zeros(len(self.v0))
        a = init(M, C, K, d_force, u, v)
        # add to results initial conditions
        self.u.append(u)
        self.v.append(v)
        self.a.append(a)

        # time
        self.time = time

        # combined stiffness matrix
        K_till = K + C.dot(gamma / (beta * t_step)) + M.dot(1 / (beta * t_step ** 2))
        inv_K_till = splu(K_till.tocsc())

        # define progress bar
        pbar = tqdm(total=int(len(self.time) - 1), unit_scale=True, unit_divisor=1000, unit="steps")

        # iterate for each time step
        for t in range(1, len(self.time)):
            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = v.dot(1 / (beta * t_step)) + a.dot(1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v.dot(gamma / beta) + a.dot(t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # update external force
            d_force = F[:, t] - F[:, t - 1]

            # external force
            force_ext = d_force.toarray()[:, 0] + m_part + c_part - np.transpose(F_abs.tocsr()) * dv

            # solve
            # du = spsolve(K_till, force_ext)
            du = inv_K_till.solve(force_ext)

            # velocity calculated through Newmark relation
            dv = du.dot(gamma / (beta * t_step)) - v.dot(gamma / beta) + a.dot(t_step * (1 - gamma / (2 * beta)))
            # acceleration calculated through Newmark relation
            da = du.dot(1 / (beta * t_step ** 2)) - v.dot(1 / (beta * t_step)) - a.dot(1 / (2 * beta))

            # update variables
            u = u + du
            v = v + dv
            a = a + da

            # add to results
            self.u.append(u)
            self.v.append(v)
            self.a.append(a)

        # convert to numpy arrays
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        self.a = np.array(self.a)

        # close the progress bar
        pbar.close()
        return

    def newmark_iterative(self, settings, M, C, K, F, F_abs, t_step, time):
        """
        Newmark integration scheme.
        Incremental formulation.
        With Newton-Raphson iterative method.

        :param settings: dictionary with the integration settings
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param F_abs: Absorbing boundary force matrix
        :param t_step: time step for the analysis
        :param time: time array for the analysis
        :return:
        """

        # constants for the Newmark integration
        beta = settings["Newmark_beta"]
        gamma = settings["Newmark_gamma"]

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, 0].toarray()
        d_force = d_force[:, 0]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        dv = np.zeros(len(self.v0))
        a = init(M, C, K, d_force, u, v)
        # add to results initial conditions
        self.u.append(u)
        self.v.append(v)
        self.a.append(a)

        # time
        self.time = time

        # combined stiffness matrix
        K_till = K + C.dot(gamma / (beta * t_step)) + M.dot(1 / (beta * t_step ** 2))
        inv_K_till = splu(K_till.tocsc())

        # define progress bar
        pbar = tqdm(total=int(len(self.time) - 1), unit_scale=True, unit_divisor=1000, unit="steps")

        # iterate for each time step newmark solver with newton raphson
        for t in range(1, len(self.time)):
            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = v.dot(1 / (beta * t_step)) + a.dot(1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v.dot(gamma / beta) + a.dot(t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # update external force
            d_force = F[:, t] - F[:, t - 1]
            d_force = d_force.toarray()[:, 0]

            # set ext force from previous time iteration
            force_ext_previous = d_force + m_part + c_part - np.transpose(F_abs.tocsr()) * dv

            # Newton-Raphson iterative method
            # initial guess
            du = np.zeros(len(self.u0))
            # tolerance
            tol = 1e-6
            # maximum number of iterations
            max_iter = 100

            # initialise
            du_tot = 0
            iter = 0
            force_previous = 0
            converged = False
            # iterate until convergence
            while not converged and iter < max_iter:
                #TODO: update force

                # external force
                force_ext = d_force + m_part + c_part - np.transpose(F_abs.tocsr()) * dv

                # update solution
                du = inv_K_till.solve(force_ext - force_previous)
                # set du for first iteration
                if iter == 0:
                    du_ini = np.copy(du)

                # energy converge criterion according to bath 1996, chapter 8.4.4
                error = np.linalg.norm(du * force_ext) / np.linalg.norm(du_ini * force_ext_previous)
                converged = (error < tol)

                # calculate total du for current time step
                du_tot += du

                # velocity calculated through Newmark relation
                dv = (du_tot * (gamma / (beta * t_step))
                      - v * (gamma / beta)
                      + a * (t_step * (1 - gamma / (2 * beta))))

                # check convergence
                if not converged:
                    force_previous = np.copy(force_ext)

                # update iteration counter
                iter += 1

            # velocity calculated through Newmark relation
            # dv = du.dot(gamma / (beta * t_step)) - v.dot(gamma / beta) + a.dot(t_step * (1 - gamma / (2 * beta)))
            # acceleration calculated through Newmark relation
            da = du_tot.dot(1 / (beta * t_step ** 2)) - v.dot(1 / (beta * t_step)) - a.dot(1 / (2 * beta))

            # update variables
            u = u + du_tot
            v = v + dv
            a = a + da

            # add to results
            self.u.append(u)
            self.v.append(v)
            self.a.append(a)


        # convert to numpy arrays
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        self.a = np.array(self.a)

        # close the progress bar
        pbar.close()
        return
