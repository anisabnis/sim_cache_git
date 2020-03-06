# sara riva project

# import

import random
import matplotlib
import matplotlib.pyplot as pl
import numpy as np
import scipy
from scipy.spatial import distance
from mpl_toolkits.mplot3d import Axes3D
import math
from itertools import cycle
import matplotlib.cm as cm
import sys

sys.setrecursionlimit(10000)
pl.switch_backend("TkAgg")
cycol = cycle('bgrcmk')
pl.style.use('seaborn')  # pretty matplotlib plots
pl.rcParams['figure.figsize'] = (12, 8)



# function

# check of parameters
def check_param(c, d, p, l, it, up):
    while True:
        # the dimension of cache must be >0
        try:
            c = int(c)
        except ValueError:
            print("the dimension of cache is not valid! Insert a new one")
            c = input()
            continue
        else:
            if c > 0:
                break
            else:
                while c <= 0:
                    print("the dimension of cache is not valid! Insert a new one")
                    c = input()
                    c = int(c)
    while True:
        # number of distributions must be >0
        try:
            d = int(d)
        except ValueError:
            print("the number of distributions is not valid! Insert a new one")
            d = input()
            continue
        else:
            if d > 0:
                break
            else:
                while d <= 0:
                    print("the number of distributions is not valid! Insert a new one")
                    d = input()
                    d = int(d)
    while True:
        # number of parameters must be >0
        try:
            p = int(p)
        except ValueError:
            print("the number of parameters is not valid! Insert a new one")
            p = input()
            continue
        else:
            if p > 0:
                break
            else:
                while p <= 0:
                    print("the number of parameters is not valid! Insert a new one")
                    p = input()
                    p = int(p)
    while True:
        # it is possibile use two different learning rate
        l = int(l)
        if l != 1 and l != 2:
            print("which learning rate insert 1 for 1/2M, or 2 for 1/(M/k)")
            l = input()
            l = int(l)
        else:
            break
    while True:
        # number of iterations must be >0
        try:
            it = int(it)
        except ValueError:
            print("the number of iterations is not valid! Insert a new one")
            it = input()
            continue
        else:
            if it > 0:
                break
            else:
                while it <= 0:
                    print("the number of iterations is not valid! Insert a new one")
                    it = input()
                    it = int(it)
    while True:
        # rate of update must be >0
        # if = 1 ... continue update
        try:
            up = int(up)
        except ValueError:
            print("the rate of update is not valid! Insert a new one")
            up = input()
            continue
        else:
            if up > 0:
                break
            else:
                while up <= 0:
                    print("the rate of update is not valid! Insert a new one")
                    up = input()
                    up = int(up)
    return c, d, p, l, it, up


# random begin state of cache
def new_cache(min, max, many, d):
    points = []
    for x in range(0, many):
        point = []
        for i in range(0, d):
            point.append(np.random.uniform(min, max))
        points.append(point)
    return points


# random mean value for a distribution of n var
def new_random_mu(min, max, n):
    mu = []
    for i in range(0, n):
        mu.append(np.random.uniform(min, max))
    return mu


# new random covar
def new_random_cov(n):
    if n == 1:
        return random.uniform(0.5, 10)
    # random matrix generation
    A = []
    for i in range(0, n):
        A.append([])
        for j in range(0, n):
            A[i].append([])
    for i in range(0, n):
        for j in range(0, n):
            A[i][j] = random.triangular(-1, 1, 0)
    A = np.array(A)
    B = np.matmul(A, A.transpose())
    C = np.zeros((n, n))
    for i in range(0, n):
        diago = 1 / math.sqrt(B[i][i])
        C[i][i] = diago
    first_mol = np.matmul(C, B)
    C = np.matmul(first_mol, C)
    D = np.zeros((n, n))
    for i in range(0, n):
        diago = random.uniform(0.5, 10)
        D[i][i] = diago
    first_mol = np.matmul(D, C)
    res = np.matmul(first_mol, D)
    return res


# sample of the distribution according with roulette method and then sample of a point with uniform prob
def random_point_iter(cumul):
    item = [0, 0]
    for x in range(0, number_of_distributions):
        rand = round(random.uniform(0, 1), 1)
        for y in range(0, number_of_distributions):
            if cumul[y] >= rand:
                # sample of the point inside the distribution
                item = random.choice(data_d[y])
                break
    return item


# print new iteration
def iteration(cache, cumul):
    item = random_point_iter(cumul)
    # pl.scatter(cache[:, 0], cache[:, 1], c='black')
    # pl.scatter([item[0]], [item[1]], c='blue')
    # pl.show()
    return item


# compute the gradient of ||y-x||^2
def derivative_function(costant, point):
    if isinstance(costant, list):
        vett = []
        dim = len(costant)
        for i in range(0, dim):
            vett.append(2 * (point[i] - costant[i]))
        return vett
    else:
        # one parameter case
        return 2 * (point - costant)


# calculate step
def calculate_new_pos(newpoint, to_be_agg, dim, l):
    alfa = 1
    if l == 1:
        # stima pessimistica
        alfa = 1 / (2 * (distance.euclidean([-10, 10], [10, 10])))
    if l == 2:
        # bigger
        alfa = 1 / ((distance.euclidean([-10, 10], [10, 10])) / dim)
    # gradient
    vett = derivative_function(newpoint, to_be_agg)
    vett = np.array(vett)
    # sgd
    pox = to_be_agg - (alfa * vett)
    return pox


# repeat with diff learning rate
def repeat_diff_l(cacheb, backup_point, learning_rate_selected, number_of_request, cumul, number_of_parameters):
    if learning_rate_selected == 1:
        learning_rate_selected = 2
    else:
        learning_rate_selected = 1
    counter_iterat = 0
    error_avg = 0
    past_error = []
    for i in range(0, number_of_request):
        # begin new iteration
        counter_iterat = counter_iterat + 1
        newpoint = backup_point[i]
        # compute distances
        min = sys.float_info.max
        to_be_agg = -1
        for x in range(0, len(cacheb)):
            c_point = cacheb[x]
            if number_of_parameters == 2:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2))
            if number_of_parameters == 3:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2) + (
                        (c_point[2] - newpoint[2]) ** 2))
            if number_of_parameters > 3:
                dist = distance.euclidean(c_point, newpoint)
            if dist <= min:
                min = dist
                to_be_agg = x
        error_avg = error_avg + min
        past_error.append(error_avg / counter_iterat)
        print("query point:", newpoint)
        print("candidate point:")
        # plot_selected(cache, newpoint, to_be_agg)
        change = to_be_agg
        to_be_agg = cacheb[to_be_agg]
        print(to_be_agg)
        print("the prediction is the label of this point and the error is:")
        print(min)
        new_pos = calculate_new_pos(newpoint, to_be_agg, dim_cache, learning_rate_selected)
        # plot_agg(cache, newpoint, to_be_agg, new_pos)
        # aggiornamento della posizione a ogni iterazione
        cacheb[change] = new_pos
    return counter_iterat, error_avg, past_error, cacheb


# end functions

# main

# parameters of execution
rand_domain_min = -20
rand_domain_max = 20
print("capacity of cache? insert the number of possible points")
c = input()
print("how many distribution?")
d = input()
print("how many parameters, insert 2 or 3")
p = input()
print("which learning rate insert 1 for 1/2M, or 2 for 1/(M/k)")
l = input()
print("how many iterations must be simulated?")
it = input()
print("rate for cache update? in number of iterations")
up = input()
# check of feasible parameter
c, d, p, l, it, up = check_param(c, d, p, l, it, up)
dim_cache = int(c)
number_of_distributions = int(d)
number_of_parameters = int(p)
learning_rate_selected = int(l)
number_of_request = int(it)
update_rate = int(up)

# fix domain of mean of variables
domain_min = -20
domain_max = 20

# memorize the error on average
error_avg = 0
# number of iteration counter
counter_iteration = 0
# story of error in different iteration
past_error = []

# the mean in multidimensional case is a vector, otherwise is a value
# random generation of means for every distribution
average_vector = []
for x in range(0, number_of_distributions):
    average_vector.append(new_random_mu(domain_min, domain_max, number_of_parameters))

# correlation
cov_matrix = []
for x in range(0, number_of_distributions):
    cov_matrix.append(new_random_cov(number_of_parameters))

# random data
data_d = []
for x in range(0, number_of_distributions):
    if number_of_parameters == 1:
        data_d.append(np.random.normal(average_vector[x], cov_matrix[x], 1000))
    else:
        data_d.append(np.random.multivariate_normal(average_vector[x], cov_matrix[x], 1000))

# generate probabilities roulette wheel
prob = []
for x in range(0, number_of_distributions):
    prob.append(round(random.uniform(0.01, 1), 1))
sum_pesi = sum(prob)
for x in range(0, number_of_distributions):
    prob[x] = prob[x] / sum_pesi
cumul = [prob[0]]
for x in range(1, number_of_distributions):
    cumul.append(cumul[x - 1] + prob[x])

# plot data
if number_of_parameters == 1:
    for x in range(0, number_of_distributions):
        # this plot produce a warning
        count, bins, ignored = pl.hist(data_d[x], 30, normed=True)
        pl.plot(bins, 1 / (cov_matrix[x] * np.sqrt(2 * np.pi)) * np.exp(
            - (bins - average_vector[x]) ** 2 / (2 * cov_matrix[x] ** 2)), linewidth=2,
                color=next(cycol))
    pl.title('Random generated points of Gaussian distributions')
    pl.show()
if number_of_parameters == 2:
    for x in range(0, number_of_distributions):
        color = list(np.random.choice(range(256), size=3))
        color = [color]
        data = data_d[x]
        pl.scatter(data[:, 0], data[:, 1], c=next(cycol))
    pl.title('Random generated points of Gaussian distributions')
    pl.show()
if number_of_parameters == 3:
    x = np.arange(10)
    ys = [i + x + (i * x) ** 2 for i in range(10)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    # Create plot
    fig = pl.figure()
    ax = fig.gca(projection='3d')

    for i in range(0, number_of_distributions):
        data_i = data_d[i]
        x, y, z = data_i[:, 0], data_i[:, 1], data_i[:, 2]
        ax.scatter(x, y, z, alpha=0.8, edgecolors='none')

    pl.title('Random distributions generated')
    pl.show()

# initialize the cache in according to the parameters
cache = new_cache(rand_domain_min, rand_domain_max, dim_cache, number_of_parameters)
cache = np.array(cache)

# plot cache
if number_of_parameters == 2:
    pl.scatter(cache[:, 0], cache[:, 1], c='black')
    pl.title('Initial cache')
    pl.show()
if number_of_parameters == 3:
    # Create plot
    fig = pl.figure()
    ax = fig.gca(projection='3d')
    x, y, z = cache[:, 0], cache[:, 1], cache[:, 2]
    ax.scatter(x, y, z, alpha=0.8, edgecolors='none')
    pl.title('Initial cache')
    pl.show()

backup_c = cache.copy()
backup_point = []

if update_rate == 1:
    for i in range(0, number_of_request):
        # begin new iteration
        counter_iteration = counter_iteration + 1
        # sample for the new point in input to the cache
        newpoint = iteration(cache, cumul)
        # memorize all points if after we want recompute with a different learning rate
        backup_point.append(newpoint)
        # compute distances
        min = sys.float_info.max
        to_be_agg = -1
        for x in range(0, dim_cache):
            c_point = cache[x]
            if number_of_parameters == 2:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2))
            if number_of_parameters == 3:
                print(newpoint, c_point)
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2) + ((c_point[2] - newpoint[2]) ** 2))
            if number_of_parameters > 3:
                dist = distance.euclidean(c_point, newpoint)
            if dist <= min:
                min = dist
                to_be_agg = x
        # compute error on avg
        error_avg = error_avg + min
        past_error.append(error_avg / counter_iteration)
        print("query point:", newpoint)
        print("candidate point:")
        change = to_be_agg
        # the candidate point
        to_be_agg = cache[to_be_agg]
        print(to_be_agg)
        print("the prediction is the label of this point and the error is:")
        print(min)
        # update position of the point in the cache
        new_pos = calculate_new_pos(newpoint, to_be_agg, dim_cache, learning_rate_selected)
        # update at each iteration
        cache[change] = new_pos

    print("end simulation. error on average is :", error_avg / counter_iteration)

    # plot error on iterations
    x_ax = np.arange(1, number_of_request + 1, 1)
    fig, ax = pl.subplots()
    ax.plot(x_ax, past_error)
    ax.set(xlabel='# iterations', ylabel='error on average', title='error vs iterations')
    ax.grid()
    pl.show()

    # it is possible compare with the other learning rate
    print("repeat the operation with different learning rate? yes or no")
    risp = input()
    if risp == "yes":
        # the simulation restart from the begging with the same parameter and same sampled points
        second_counter_iterat, second_error_avg, second_past_error, second_cache = repeat_diff_l(backup_c, backup_point,
                                                                                                 learning_rate_selected,
                                                                                                 number_of_request,
                                                                                                 cumul,
                                                                                                 number_of_parameters)
        # plot comparison of the learning rates and the error on iterations
        x_ax = np.arange(1, number_of_request + 1, 1)
        fig, ax = pl.subplots()
        ax.plot(x_ax, past_error)
        ax.plot(x_ax, second_past_error)
        ax.set(xlabel='# iterations', ylabel='error on average', title='error vs iterations')
        ax.grid()
        pl.show()

    # plot of the resulting position in the cache only for 2-d or 3-d otherwise print of the cache
    if number_of_parameters == 2:
        backup_point = np.array(backup_point)
        pl.scatter(backup_point[:, 0], backup_point[:, 1], c="red")
        pl.scatter(cache[:, 0], cache[:, 1], c="black")
        pl.title('Resulting cache')
        pl.show()
    if number_of_parameters == 3:
        x = np.arange(10)
        ys = [i + x + (i * x) ** 2 for i in range(10)]

        colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        # Create plot
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        x, y, z = cache[:, 0], cache[:, 1], cache[:, 2]
        ax.scatter(x, y, z, alpha=0.8, color="black", edgecolors='none')
        for i in range(0, number_of_distributions):
            data_i = data_d[i]
            x, y, z = data_i[:, 0], data_i[:, 1], data_i[:, 2]
            ax.scatter(x, y, z, alpha=0.8, edgecolors='none')

        pl.title('Resulting cache')
        pl.show()
    if number_of_parameters > 3:
        print("average vector:")
        print(average_vector)
        print("inital cache:")
        print(backup_c)
        print("final cache:")
        print(cache)
        if risp == "yes":
            print("second resulting cache:")
            print(second_cache)

    #print of the roulette wheel probabilities
    print("the probabilities of distributions are:", cumul)
else:
    # compute also the rate=1 for compare (possible cache and error)
    counter_interval = 0
    possible_cache = cache.copy()
    possible_error = []
    possible_error_sum = 0
    for i in range(0, number_of_request):
        # begin new iteration
        counter_iteration = counter_iteration + 1
        counter_interval = counter_interval + 1
        newpoint = iteration(cache, cumul)
        backup_point.append(newpoint)
        # compute distances
        min = sys.float_info.max
        to_be_agg = -1
        for x in range(0, dim_cache):
            c_point = cache[x]
            if number_of_parameters == 2:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2))
            if number_of_parameters == 3:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2) + (
                        (c_point[2] - newpoint[2]) ** 2))
            if number_of_parameters > 3:
                dist = distance.euclidean(c_point, newpoint)
            if dist <= min:
                min = dist
                to_be_agg = x
        print("query point:", newpoint)
        print("candidate point:")
        # plot_selected(cache, newpoint, to_be_agg)
        change = to_be_agg
        to_be_agg = cache[to_be_agg]
        print(to_be_agg)
        print("the prediction is the label of this point and the error is:")
        print(min)
        new_pos = calculate_new_pos(newpoint, to_be_agg, dim_cache, learning_rate_selected)
        # plot_agg(cache, newpoint, to_be_agg, new_pos)
        error_avg = error_avg + min
        past_error.append(error_avg / counter_iteration)
        # update in according to update rate
        if counter_interval == update_rate:
            cache[change] = new_pos
            counter_interval = 0

        # computation with rate =1 (same procedure)
        min = sys.float_info.max
        to_be_agg = -1
        for x in range(0, dim_cache):
            c_point = possible_cache[x]
            if number_of_parameters == 2:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2))
            if number_of_parameters == 3:
                dist = math.sqrt(((c_point[0] - newpoint[0]) ** 2) + ((c_point[1] - newpoint[1]) ** 2) + (
                        (c_point[2] - newpoint[2]) ** 2))
            if number_of_parameters > 3:
                dist = distance.euclidean(c_point, newpoint)
            if dist <= min:
                min = dist
                to_be_agg = x
        change = to_be_agg
        possible_error_sum = possible_error_sum + min
        to_be_agg = possible_cache[change]
        new_pos = calculate_new_pos(newpoint, to_be_agg, dim_cache, learning_rate_selected)
        possible_error.append(possible_error_sum / counter_iteration)
        possible_cache[change] = new_pos

    # plot of error on iterations of the update rate != 1
    x_ax = np.arange(1, number_of_request + 1, 1)
    fig, ax = pl.subplots()
    ax.plot(x_ax, past_error)
    ax.set(xlabel='# iterations', ylabel='error on average', title='error vs iterations')
    ax.grid()
    pl.show()

    # plot of the error also with rate=1
    x_ax = np.arange(1, number_of_request + 1, 1)
    fig, ax = pl.subplots()
    ax.plot(x_ax, past_error, label="error of rate", color="black")
    ax.plot(x_ax, possible_error, label="always update cache", color="green")
    ax.set(xlabel='# iterations', ylabel='error on average', title='error vs iterations')
    ax.grid()
    pl.show()

    # plot of the resulting position of the both cache for 2-d or 3-d otherwise print of the two cache
    if number_of_parameters == 2:
        backup_point = np.array(backup_point)
        pl.scatter(backup_point[:, 0], backup_point[:, 1], c="red")
        pl.scatter(cache[:, 0], cache[:, 1], c="black")
        pl.title('Resulting cache')
        pl.show()

        backup_point = np.array(backup_point)
        pl.scatter(backup_point[:, 0], backup_point[:, 1], c="red")
        pl.scatter(cache[:, 0], cache[:, 1], c="black")
        pl.scatter(possible_cache[:, 0], possible_cache[:, 1], c="orange")
        pl.title('Resulting caches')
        pl.show()
    if number_of_parameters == 3:
        x = np.arange(10)
        ys = [i + x + (i * x) ** 2 for i in range(10)]

        colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        # Create plot
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        x, y, z = cache[:, 0], cache[:, 1], cache[:, 2]
        ax.scatter(x, y, z, alpha=0.8, color="black", edgecolors='none')
        for i in range(0, number_of_distributions):
            data_i = data_d[i]
            x, y, z = data_i[:, 0], data_i[:, 1], data_i[:, 2]
            ax.scatter(x, y, z, alpha=0.8, edgecolors='none')

        pl.title('Resulting cache')
        pl.show()

        # Create plot
        fig = pl.figure()
        ax = fig.gca(projection='3d')
        x, y, z = possible_cache[:, 0], possible_cache[:, 1], possible_cache[:, 2]
        ax.scatter(x, y, z, alpha=0.8, color="orange", edgecolors='none')
        x, y, z = cache[:, 0], cache[:, 1], cache[:, 2]
        ax.scatter(x, y, z, alpha=0.8, color="black", edgecolors='none')
        for i in range(0, number_of_distributions):
            data_i = data_d[i]
            x, y, z = data_i[:, 0], data_i[:, 1], data_i[:, 2]
            ax.scatter(x, y, z, alpha=0.8, edgecolors='none')

        pl.title('Resulting caches')
        pl.show()
    if number_of_parameters > 3:
        print("average vector:")
        print(average_vector)
        print("inital cache:")
        print(backup_c)
        print("final cache:")
        print(cache)
        print("cache with continuous update:")
        print(possible_cache)

    # print of the roulette wheel probabilities
    print("the probabilities of distributions are:", cumul)

print("end simulation.")
