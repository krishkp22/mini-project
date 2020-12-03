import numpy as np
import scipy.interpolate as sc
import matplotlib.pyplot as plt
import three_link
import invkin
import timeit




def generate_trajectories(formatted_population, start, end, fitness_calculated):
    shape = np.shape(formatted_population)
    left_end, right_end = start, start
    if start[0] < end[0]:
        right_end = end
    else:
        left_end = end

    population_trajectories = [False for g in range(shape[0])]
    trajectory_points = np.zeros([shape[0], shape[1] + 2, shape[2]])
    for i in range(shape[0]):
        if fitness_calculated[i]:
            continue
        ch_with_start = np.insert(formatted_population[i, :, :], 0, left_end, axis=0)
        chrome_all_pts = np.insert(ch_with_start, (shape[1] + 1), right_end, axis=0)
        population_trajectories[i] = sc.PchipInterpolator(chrome_all_pts[:, 0], chrome_all_pts[:, 1])
        trajectory_points[i, :, :] = chrome_all_pts
    return trajectory_points, population_trajectories


def chrome_traj(chrome, start, end):
    sorted_chrome = format(chrome)
    sh = np.shape(sorted_chrome)
    left_end, right_end = start, start
    if start[0] < end[0]:
        right_end = end
    else:
        left_end = end
    K = sh[1]
    ch_with_start = np.insert(sorted_chrome, 0, left_end, axis=1)
    chrome_all_pts = np.insert(ch_with_start, (K + 1), right_end, axis=1)
    ch_x, ch_y = chrome_all_pts[:, :, 0][0], chrome_all_pts[:, :, 1][0]
    trajectory = sc.PchipInterpolator(ch_x, ch_y)
    traj_points = path_points(trajectory, 0.1, start, end)
    return traj_points


def format(population) -> object:
    shape = np.shape(population)
    if len(shape) == 1:
        P = 1
        K = int(shape[0]/2)
    elif len(shape) == 2:
        P = shape[0]
        K = int(shape[1]/2)
    formatted_population = np.zeros([P, K, 2])
    for i in range(P):
        if P == 1:
            chrome = np.reshape(population, [K, 2])
        else:
            chrome = np.reshape(population[i, :], [K, 2])
        chrome_sorted = chrome[chrome[:, 0].argsort()].transpose()
        formatted_population[i, :, :] = chrome_sorted.transpose()
    return formatted_population


def check_point_validity(formatted_population, link_len, start, end) -> list:
    shape = np.shape(formatted_population)
    left_end, right_end = start, start
    if start[0] < end[0]:
        right_end = end
    else:
        left_end = end
    validity = []
    for i in range(shape[0]):
        r = np.linalg.norm(formatted_population[i, :, :], axis=1)
        if np.any(formatted_population[i, :, 0] < left_end[0]):
            validity.append(False)
        elif np.any(formatted_population[i, :, 0] > right_end[0]):
            validity.append(False)
        elif np.all(r > link_len[0]):
            if np.all(r < (sum(link_len))):
                if np.all(formatted_population[i,:,1] > 0):
                    validity.append(True)
                else:
                    validity.append(False)
            else:
                validity.append(False)
        else:
            validity.append(False)

    return validity


def check_trajectory_validity(trajectory, obstacles):
    obstacles = np.array(obstacles)

    if np.any(trajectory(obstacles[:,0]) > obstacles[:,1]):
        validity = False
    else:
        validity = True
    return validity


def path_points(y, epsilon, start, end):
    pt_x = [start[0]]
    pt_y = [start[1]]
    der = y.derivative()

    x = start[0]

    if start[0] < end[0]:  
        while x < end[0]:
            del_x = epsilon / np.sqrt(der(x) ** 2 + 1)
            if (x + del_x) < end[0]:
                pt_x.append(x + del_x)
                pt_y.append(y(x + del_x))
                x += del_x
            else:
                pt_x.append(end[0])
                pt_y.append(end[1])
                break
    else:  # end point on left side
        while x > end[0]:
            del_x = epsilon / np.sqrt(der(x) ** 2 + 1)
            if (x - del_x) > end[0]:
                pt_x.append(x - del_x)
                pt_y.append(y(x - del_x))
                x -= del_x
            else:
                pt_x.append(end[0])
                pt_y.append(end[1])
                break

    points = np.zeros([2, len(pt_x)])
    points[0, :] = np.array(pt_x)
    points[1, :] = np.array(pt_y)

    return points.transpose()


def fitness_population(population, link_len, start_pt, end_pt, obstacles, epsilon, mu, Single=False):
    """
    Envelope function for complete fitness calculation
    Order of operations:
    1. point checking       (set fitness to zero for invalid)
    2. path interpolation
    3. path discretization
    4. reverse kinematics on path
    5. Path checking        (check order here)
    5. fitness calculation
    """
    if len(link_len) == 3:
        arm1 = three_link.Arm3Link(link_len)
    elif len(link_len) == 2:
        arm1 = invkin.Arm(link_len)

    if Single == True:
        pop_size = 1
    else:
        pop_size = np.shape(population)[0]

    cost_pop = [np.inf for i in range(pop_size)]  
    fitness_calculated = [False for i in range(pop_size)]  

    formatted_pop = format(population)
    pt_validity = check_point_validity(formatted_pop, link_len, start_pt, end_pt)
    for i in range(pop_size):
        if pt_validity[i] == False:
            cost_pop[i] = np.inf
            fitness_calculated[i] = True

    points, trajectories = generate_trajectories(formatted_pop, start_pt, end_pt, fitness_calculated)
    traj_points = None
    for i in range(pop_size):
        if fitness_calculated[i] == False:
            traj_points = path_points(trajectories[i], epsilon, start_pt, end_pt)
            theta = np.array(arm1.time_series(traj_points))
            validity = check_trajectory_validity(trajectories[i], obstacles)
            if validity == False:
                cost_pop[i] = np.inf
            else:
                cost_pop[i] = fitness_chrome(theta, mu)
            fitness_calculated[i] = True

    fitness_pop = 1/np.array(cost_pop)
    return np.array(fitness_pop), traj_points


def fitness_chrome(theta, mu):

    theta = theta.T
    div = np.shape(theta)[1]
    theta_i = theta[:, 0:div - 2]
    theta_j = theta[:, 1:div - 1]
    del_theta = abs(theta_j - theta_i)
    fitness = 0
    for i in range(div - 2):
        for j in range(len(mu)):
            fitness += mu[j] * theta[j, i]
    return fitness
