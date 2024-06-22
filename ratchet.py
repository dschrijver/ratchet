import numpy as np
from numba import njit
import matplotlib.pyplot as plt


@njit(cache=True) # dM/dt = f
def f(t, y):
    dM = np.zeros(y.shape, dtype=np.float64)
    # The loop excludes the outer two layers, effectively keeping the magnetizations fixed. 
    # This could be reproduced in experiment by adding antiferromagnetic materials
    # which fix the outer layers. 
    for i in range(1, N_layers-1):
        M = y[i]
        H_eff = H_ext_dir*H_ext(t) - N@M + \
                2*K_list[i]/(mu_0*M_s*M_s)*M[2]*np.array([0.0, 0.0, 1.0], dtype=np.float64)
        H_eff += J_list[i-1]/(mu_0*M_s*M_s*d_list[i])*y[i-1]
        H_eff += J_list[i]/(mu_0*M_s*M_s*d_list[i])*y[i+1]
        dM[i] = -gamma*mu_0/(1+alpha*alpha)*(np.cross(M, H_eff) + \
                                             alpha/M_s*np.cross(M, np.cross(M, H_eff)))
    return dM

@njit(cache=True) # Runge-Kutta 4
def rk(t, y, dt,):
    k1 = dt*f(t, y)
    k2 = dt*f(t+dt/2, y+k1/2)
    k3 = dt*f(t+dt/2, y+k2/2)
    k4 = dt*f(t+dt, y+k3)
    return y + 1/6*(k1 + 2*k2 + 2*k3 + k4)

@njit(cache=True) # Rectangular pulse external magnetic field
def H_ext(t):
    t_total = np.sum(H_ext_times)
    t %= t_total
    for i in range(len(H_ext_amps)):
        if t < np.sum(H_ext_times[:i+1]):
            return H_ext_amps[i]
        
def main():
    # --- Parameter Setup --- #
    # All variables are expressed in SI units.
    global gamma, mu_0, alpha, M_s
    gamma = 1.7609e11
    mu_0 = 1.2566e-6
    alpha = 0.05
    M_s = 1.29e6

    # Initial magnetizations
    global N_layers
    N_layers = 8 # Assumed to be even and at least 2
    M = np.empty((N_layers, 3), dtype=np.float64)
    sign = -1
    for i in range(N_layers//2):
        M[N_layers//2 + i] = np.array([0.0, 0.0, sign*M_s], dtype = np.float64)
        M[N_layers//2-1 - i] = np.array([0.0, 0.0, sign*M_s], dtype = np.float64)
        sign *= -1
    
    # Demag field
    global N
    N_x = 0.0
    N_y = 0.0
    N_z = 1-N_x-N_y
    N = np.diag([N_x, N_y, N_z])

    # Layer thicknesses
    global d_list
    d = [0.7e-9, 0.8e-9]
    Delta_d = d[1]-d[0]
    d_list = np.empty(N_layers, dtype=np.float64)
    for i in range(N_layers//2):
        d_list[N_layers//2 + i] = d[(i)%2]
        d_list[N_layers//2-1-i] = d[(i+1)%2]

    # Uniaxial anisotropy along the z-axis
    global K_list
    K = [1.66e6, 1.59e6]
    Delta_K = K[0]-K[1]
    K_list = np.empty(N_layers, dtype=np.float64)
    for i in range(N_layers//2):
        K_list[N_layers//2 + i] = K[(i)%2]
        K_list[N_layers//2-1-i] = K[(i+1)%2]

    # General ratchet requirements
    req1 = (2*K[0] - mu_0*M_s*M_s)*d[0]
    req2 = (2*K[1] - mu_0*M_s*M_s)*d[1]
    req3 = 2*(Delta_K)*(d[0]*d[1])/(Delta_d)
    requirement = max([req1, req2, req3])

    # RKKY Exchange coupling between layers
    # The second item in the list will be the coupling between
    # starting soliton in the middle.
    global J_list
    J = [-0.00122, -0.00034]
    Delta_J = abs(J[0])-abs(J[1])
    if (Delta_J < requirement):
        print("The difference between J values ( now ", Delta_J, ") must be larger than", requirement)
    J_list = [J[1]]
    for i in range((N_layers-2)//2):
        J_list = [J[i % 2]] + J_list + [J[i % 2]]
    J_list = np.array(J_list)

    # External magnetic field
    global H_ext_dir, H_ext_amps, H_ext_times, N_ratchet_steps
    theta_H = 0.00095
    eps_1 = 0.8
    eps_2 = 1.0
    H_ext_dir = np.array([np.sin(theta_H), 0, np.cos(theta_H)], dtype=np.float64)
    H_1 = 2*K[1]/(mu_0*M_s) - M_s + Delta_J/(mu_0*M_s*d[1]) + eps_1*((2/(mu_0*M_s)*(Delta_K)) + \
          Delta_J/(mu_0*M_s)*(Delta_d)/(d[0]*d[1]))
    H_2 = -2*K[1]/(mu_0*M_s) + M_s + Delta_J/(mu_0*M_s*d[1]) - eps_2*((-2/(mu_0*M_s)*(Delta_K)) + \
          Delta_J/(mu_0*M_s)*(Delta_d)/(d[0]*d[1]))
    H_ext_amps = np.array([H_1, H_2])
    H_ext_times = np.array([1.5e-9, 1.5e-9]) # Must have the same length as H_ext_amps
    N_ratchet_steps = 2

    # Iteration parameters
    t_final = (N_ratchet_steps//2)*np.sum(H_ext_times)+(N_ratchet_steps%2)*H_ext_times[0]
    N_steps = 1_000_000
    dt = t_final/(N_steps-1)
    t_list = np.linspace(0, t_final, N_steps)

    # --- Evolution of magnetizations --- #
    M_list = np.empty((N_steps, N_layers, 3), dtype=np.float64)
    for i in range(len(t_list)):
        t = t_list[i]
        M_list[i] = M
        M = rk(t, M, dt)
        if i % 1000 == 0:
            print("Progress:", round((i+1)/len(t_list)*100, 2), "%", end="\r")

    # --- Plotting --- #
    legend_list = []
    for n in range(N_layers):
        color = "#%06x" % np.random.randint(0, 0xFFFFFF)
        plt.plot(t_list[0::100]*1e9, M_list[:, n, 2][0::100]/M_s + np.full(t_list.shape, -(1+2.5*n) + (2+2.5*(N_layers-1)))[0::100], color=color)
        plt.axhline(-2.5*n + (2+2.5*(N_layers-1)), color=color, ls="--", lw=1)
        plt.axhline(-(2+2.5*n) + (2+2.5*(N_layers-1)), color=color, ls="--", lw=1)
        legend_list.extend(["Layer " + str(n+1), "_", "_"])

    i = 0
    t = H_ext_times[i]
    while t < t_final:
        plt.axvline(t*1e9, color="black", ls="--", lw=1)
        i += 1
        t += H_ext_times[i % len(H_ext_times)]
    legend_list.extend(i*["_"])

    plt.xlabel("t (ns)")
    plt.ylabel("$M_z/M_s$")
    plt.legend(legend_list)
    plt.title("Magnetic ratchet of " + str(N_layers) + " layers")
    plt.title("$J_1 = "+str(J[0])+"$, $J_2 = "+str(J[1])+"$")
    plt.show()

    plt.plot(t_list[::100]*1e9, [H_ext(t) for t in t_list[::100]])
    plt.xlabel("t (ns)")
    plt.ylabel("$|\\vec{H_{ext}}|$ (A/m)")
    plt.axhline(0, color="black", ls="--", lw=1)
    plt.show()



if __name__ == "__main__":
    main()