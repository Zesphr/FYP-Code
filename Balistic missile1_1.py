import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.optimize as spo
from pyswarm import pso
from geneticalgorithm2 import geneticalgorithm2 as ga
from matplotlib.lines import Line2D
import time

def ode(t, states, alpha, thrust, mass0):
    # Constants
    S_body = np.pi * 0.26 ** 2
    # S_wings = 1.011
    g0 = 9.81
    Isp = 263.1

    # States
    gamma = states[0]
    velo = states[1]
    x = states[2]
    y = states[3]
    massf = states[4]


    # Aerodynamic co-efficients
    c1 = 3.312e-9
    c2 = -1.142e-4
    c3 = 1.224

    cnb = 0.0432 * alpha + 0.1277
    cab = -0.0002 * alpha ** 2 + 0.0058 * alpha + 0.3042

    # cnw = -0.0003*alpha**3 + 0.0019*alpha**2 + 0.1046*alpha - 0.0763
    # caw = 9E-08*alpha**6 + 5E-07*alpha**5 - 3E-06*alpha**4 - 8E-05*alpha**3 - 0.0006*alpha**2 - 0.0005*alpha + 0.0343


    # https://sci-hub.wf/10.1002/oca.807
    xdot = velo * np.cos(gamma)
    ydot = velo * np.sin(gamma)
    mdot = - thrust / (Isp * g0)

    mass = mass0 + massf

    rho = c1 * y ** 2 + c2 * y + c3
    q = 0.5 * rho * (velo ** 2)
    normal = (q * S_body * cnb) # + (q * S_wings * cnw)   # (q * S * Cl)
    axial = (q * S_body * cab) # + (q * S_wings * caw)    # (q * S * Cd)

    alpha_rad = np.radians(alpha)

    gammadot = (((thrust - axial) / (mass * velo)) * np.sin(alpha_rad)) + ((normal / (mass * velo)) * np.cos(alpha_rad)) - ((g0 * np.cos(gamma)) / velo)
    Vdot = (((thrust - axial) / mass) * np.cos(alpha_rad)) - ((normal / mass) * np.sin(alpha_rad)) - (g0 * np.sin(gamma))



    out = [gammadot, Vdot, xdot, ydot, mdot]
    # print(out, t)
    return out


def trajectory(alpha, plot_ops, ti, tf, thrust, mass0, vals):

    def ground(t, states, alpha, thrust, mass0):
        return states[3]
    ground.terminal = True


    time = np.linspace(ti, tf, 1000)
    sol1 = solve_ivp(ode, [ti, tf], y0=vals, t_eval=time, method='RK45',
                     args=(alpha, thrust, mass0), atol=1e-7, rtol=1e-4, events=ground, max_step=0.5)

    try:
        impact = sol1.y_events[0][0][2]
        # print("Impact at ", impact)
    except IndexError:
        impact = False  #sol1.y[2][-1]
        # print("OUTATIME", impact)

    if plot_ops[0] is True:

        ##### DO NOT PLT FIGURE 6, IT IS RESERVED FOR COST FUNCTION

        plt.figure(1)
        plt.plot(sol1.y[2], sol1.y[3], plot_ops[1], ls=plot_ops[2])
        # plt.rcParams.update({'font.size': fontsize})
        plt.xlabel('Down Range (m)')
        plt.ylabel('Altitude (m)')
        plt.title('Trajectory Profile - Ballistic Missile')
        line1 = [Line2D([0], [0], color='black', lw=2, linestyle='-.'),
                 Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                 Line2D([0], [0], color='red', lw=2, linestyle=':')]
        plt.legend(line1, ['SLSQP', 'Genetic', 'PSO'])
        plt.axis('equal')

        plt.figure(2)
        plt.plot(sol1.t, sol1.y[3], plot_ops[1], ls=plot_ops[2])
        plt.xlabel('Time (t)')
        plt.ylabel('Altitude (m)')
        plt.title('Vertical Profile - Ballistic Missile')
        line2 = [Line2D([0], [0], color='black', lw=2, linestyle='-.'),
                 Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                 Line2D([0], [0], color='red', lw=2, linestyle=':')]
        plt.legend(line2, ['SLSQP', 'Genetic', 'PSO'])
        #
        plt.figure(3)
        plt.plot(sol1.t, sol1.y[1], plot_ops[1], ls=plot_ops[2])
        plt.xlabel('Time (t)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Velocity Variation - Ballistic Missile')
        line3 = [Line2D([0], [0], color='black', lw=2, linestyle='-.'),
                 Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                 Line2D([0], [0], color='red', lw=2, linestyle=':')]
        plt.legend(line3, ['SLSQP', 'Genetic', 'PSO'])
        #
        plt.figure(4)
        plt.plot(sol1.t, sol1.y[4], plot_ops[1], ls=plot_ops[2])
        plt.xlabel('Time (t)')
        plt.ylabel('Mass (kg)')
        plt.title('Mass Loss Due to Fuel - Ballistic Missile')
        line4 = [Line2D([0], [0], color='black', lw=2, linestyle='-.'),
                 Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                 Line2D([0], [0], color='red', lw=2, linestyle=':')]
        plt.legend(line4, ['SLSQP', 'Genetic', 'PSO'])

        plt.figure(5)
        plt.xlabel('Time (t)')
        plt.ylabel('Angle of Attack (degrees)')
        plt.title('Angle of Attack Profile - Ballistic Missile')
        plt.hlines(y=alpha, xmin=ti, xmax=tf, colors=plot_ops[1], ls=plot_ops[2])
        line5 = [Line2D([0], [0], color='black', lw=2, linestyle='-.'),
                 Line2D([0], [0], color='blue', lw=2, linestyle='-'),
                 Line2D([0], [0], color='red', lw=2, linestyle=':')]
        plt.legend(line5, ['SLSQP', 'Genetic', 'PSO'], loc='center right')



    return [impact, sol1.y[0][-1], sol1.y[1][-1], sol1.y[2][-1], sol1.y[3], sol1.y[4][-1]]

def single_shooting(param, plot=False, colour='b', style='-'):

    plot_ops = [plot, colour, style]

    gamma = np.radians(40)  # 40
    velo = 1.0
    x_pos = 0.0
    y_pos = 0.001
    mass_f = 520

    t_int = 0.0
    t_step = 10.0
    t_final = t_int + t_step

    phase = 0

    heights = []
    velos = []
    angles = []
    x_dist = []
    aoas = []
    max_heights = []


    heights.append(y_pos)
    velos.append(velo)
    angles.append(gamma)
    x_dist.append(x_pos)

    while True:
        mass0 = 1000

        if phase == 2:
            mass0 = mass0 - 100

        if phase < len(param) - 2:
            aoa = param[phase]
        else:
            aoa = param[-3]

        if phase <= 1 or 3 <= phase <= 4:
            thrust = 31000
        else:
            thrust = 0.0


        vals_i = [gamma, velo, x_pos, y_pos, mass_f]



        vals_f = trajectory(aoa, plot_ops, t_int, t_final, thrust, mass0, vals_i)

        gamma = vals_f[1]
        velo = vals_f[2]
        x_pos = vals_f[3]
        y_pos = vals_f[4][-1]
        mass_f = vals_f[5]

        aoas.append(aoa)
        heights.append(y_pos)
        velos.append(velo)
        angles.append(gamma)
        x_dist.append(x_pos)
        max_heights.append(max(vals_f[4]))

        if vals_f[0]:

            vals_total.append(1)
            if colour == 'b':
                vals_Gen.append(vals_f[0])
            elif colour == 'r':
                vals_PSO.append(vals_f[0])
            elif colour == 'k':
                vals_SLSQP.append(vals_f[0])
                print('iteration complete', vals_f[0])

            if plot is True:
                print('Optimisation complete', vals_f[0], max(max_heights), velos[-1], np.degrees(angles[-1]))

            # plt.show()
            break

        t_int = t_int + t_step
        t_final = t_final + t_step
        phase += 1


    return vals_f[0], max(max_heights), velos[-1], angles[-1]



def objective_function(params, plot=False, colour='b', style='-'):
    # pen_height = 0
    # pen_velo = 0
    # pen_angle = 0
    #
    # height_max = 1000
    # impact_velo = 150
    # impact_angle = 40

    results = single_shooting(params, plot, colour, style)

    #
    # if results[1] > height_max:
    #     pen_height = ((results[1] - height_max) ** 2) + (results[1] - height_max)
    #
    # if results[2] < impact_velo:
    #     pen_velo = ((impact_velo - results[2]) ** 2) * 30
    #
    # if np.degrees(results[3]) > -1 * impact_angle:
    #     pen_angle = (abs(impact_angle + np.degrees(results[3])) ** 2) * 30

    cost = -1 * results[0]  # + pen_height + pen_velo + pen_angle
    #
    if colour == 'k':
        vals_SLSQP_cost.append(cost)

    return -1 * results[0]  # + pen_height + pen_velo + pen_angle




# ----------------------------------------------------------------------------------------------------------------------


plt.figure(6)  # Easier to add a legend this way
custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', marker='.', lw=0)]
plt.legend(custom_lines, ['Genetic', 'PSO'])

vals_SLSQP = []
vals_SLSQP_cost = []
vals_PSO = []
vals_Gen = []
vals_total = []

# ----------------------------------------------------------------------------------------------------------------------
# def height_ceiling(params, plot=False, colour='b', style='-'):
#     height = 1000
#     return height - single_shooting(params)[1]
#
# def impact_velocity(params, plot=False, colour='b', style='-'):
#     v_impact = 150
#     return single_shooting(params)[2] - v_impact
#
# def impact_angle(params, plot=False, colour='b', style='-'):  # need to figure out
#     v_angle = np.radians(-40)
#     return v_angle - single_shooting(params)[3]
#
# def constraints(params, plot=False, colour='b', style='-' ):
#     height = height_ceiling(params)
#     velo = impact_velocity(params)
#     angle = impact_angle(params)
#     return [height, velo, angle]

# ----------------------------------------------------------------------------------------------------------------------

# const = (
#     {'type': 'ineq', 'fun': height_ceiling},
#     {'type': 'ineq', 'fun': impact_velocity},
#     {'type': 'ineq', 'fun': impact_angle}
# )

guess = np.array([0] * 18)
bounds = [(-10, 40)] * 18

start_time = time.time()

results = spo.minimize(objective_function, x0=guess, bounds=bounds, method='SLSQP', args=(False, 'k', '-.',), options={'maxiter': 20000})  #'Nelder-Mead', options={'maxiter': 20000, 'maxfev': 20000, 'adaptive': True})

print("--- %s seconds SLSQP ---" % (time.time() - start_time))

plt.figure(8)
lst1 = list(range(0, len(vals_SLSQP)))
plt.scatter(lst1, vals_SLSQP, c='k', marker='+', alpha=0.5)

plt.figure(8)
custom_lines = [Line2D([0], [0], color='blue', marker='*', lw=0),
                Line2D([0], [0], color='red', marker='.', lw=0),
                Line2D([0], [0], color='black', marker='+', lw=0)
                ]
plt.legend(custom_lines, ['Genetic', 'PSO', 'SLSQP'], loc='lower right')
plt.xlabel('Iteration')
plt.ylabel('Range (m)')
plt.title('Range per Function Evaluation - Ballistic Missile')
#
print(len(vals_SLSQP))

print(results)

optimum1 = single_shooting(results.x, True, 'k', '-.',)

# ----------------------------------------------------------------------------------------------------------------------

for i in range(0, len(vals_SLSQP_cost)):
    plt.figure(7)
    plt.plot(i + 1, -1 * vals_SLSQP_cost[i], 'k+',)

plt.figure(7)
custom_lines = [Line2D([0], [0], color='black', marker='+', lw=0)]
plt.legend(custom_lines, ['SLSQP'])
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('SLSQP Cost per Function Evaluation - Ballistic Missile')
# plt.show()

# ----------------------------------------------------------------------------------------------------------------------

for i in range(0,3):

    print('RUN ', i+1, '#############################################################################')

    ub = [40] * 18
    lb = [-10] * 18

    # cons = [height_ceiling]

    start_time = time.time()

    xopt, fopt = pso(objective_function, lb, ub, args=(False, 'r', ':',), swarmsize=400, maxiter=200, debug=False)

    print("--- %s seconds PSO ---" % (time.time() - start_time))
    print(len(vals_PSO))

    # -------------------------------------------------------------------------------------------------------------------

    plt.figure(8)
    lst2 = list(range(0, len(vals_PSO)))
    plt.scatter(lst2, vals_PSO, s=0.5, c='r', marker='.', alpha=0.5)

    print(xopt, fopt)

    graphs = single_shooting(xopt, True, 'r', ':')

    # ----------------------------------------------------------------------------------------------------------------------

    bnds = [(-10, 40)] * 18

    algorithm_param = {'max_num_iteration': 600,
                       'population_size': 100,
                       'mutation_probability': 0.01,
                       'elit_ratio': 0.05,
                       'parents_portion': 0.2,
                       'crossover_type': 'uniform',
                       'selection_type': 'roulette',
                       'max_iteration_without_improv': None}

    model = ga(function=objective_function, dimension=len(bnds), variable_type='real', variable_boundaries=bnds, algorithm_parameters=algorithm_param,)
    result = model.run()

    print(len(vals_Gen))

    print(result)

    plt.figure(8)
    lst3 = list(range(0, len(vals_Gen)))
    plt.scatter(lst3, vals_Gen, s=0.5, c='b', marker='*', alpha=0.5)

    print(result.variable)
    print(result.score)

    optimum2 = single_shooting(result.variable, True, 'b', '-',)

# ----------------------------------------------------------------------------------------------------------------------

# plt.show()
