import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import pickle
from random_walk import random_walk

N_STATES = 5


def convertEpisodeToStateRep(episode):
    n_steps = len(episode)
    states_visited = np.zeros((n_steps), dtype=int)
    for i in range(n_steps):
        s,r,ns = episode[i]
        states_visited[i] = s

    #encode visited states as one-hot
    states_visited_hot = np.zeros((n_steps, N_STATES))
    states_visited_hot[np.arange(n_steps), states_visited] = 1

    y = episode[-1][1]  #get final reward
    return states_visited_hot,y

def td_update(x, y, w, lambd, alpha):
    n_steps = x.shape[0]

    e_i = 0
    delta_w = np.zeros((N_STATES))
    for i in np.arange(n_steps):
        x_i = x[i,:]
        e_i = lambd * e_i + x_i

        if i == n_steps - 1:
            delta_w_i = alpha * (y - h(w, x_i)) * e_i
        else:
            x_nxt = x[i+1,:]
            delta_w_i = alpha * (h(w,x_nxt) - h(w,x_i)) * e_i
        delta_w = delta_w + delta_w_i

    return delta_w

def createTrainingSet(n_sequences):
    world = random_walk(N_STATES)
    x_train = []
    y_train = []
    for i in range(n_sequences):
        episode = world.generateEpisode()
        x, y = convertEpisodeToStateRep(episode)
        x_train.append(x)
        y_train.append(y)
    return x_train, y_train, world

def h(w,x):
    return np.dot(w,x)


def td_seq_learning(x_train, y_train, alpha, lambd, n_seqs, world):
    i = 0
    iter = 0
    w = np.zeros((N_STATES)) + 0.5
    while True:
        x = x_train[i]
        y = y_train[i]
        delta_w = td_update(x, y, w, lambd, alpha)
        w = w + delta_w

        i += 1
        iter += 1
        # restart training set index, if at end
        if i == n_seqs:
            i = 0

        # stopping criterion
        if np.max(delta_w) < 0.001 or iter == 200:
            break

    #print(w)
    rmse = sqrt(mean_squared_error(w, world.actual_state_values))
    return rmse

x_train, y_train, world = createTrainingSet(10)

lambdas = [0.1, 0.3, 0.5, 0.7, 1]
alphas = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
n_runs = 100

all_rmses = np.zeros(( len(alphas),len(lambdas)))
for k in range(len(lambdas)):
    for j in range(len(alphas)):
        rmses = np.zeros((n_runs))
        for i in range(n_runs):
            rmses[i] = td_seq_learning(x_train, y_train, alphas[j], lambdas[k], 10, world)
        rmse = np.mean(rmses)
        all_rmses[j,k] = rmse

print(all_rmses)
pickle.dump(all_rmses, open("all_rmses.pkl", "wb"))

# all_rmses = pickle.load(open("all_rmses1.pkl", "rb"))

fig = plt.figure()
ax = fig.add_subplot(111)

for k in range(len(lambdas)):
    label = r'$\lambda =$' + str(lambdas[k])
    plt.plot(np.arange(len(alphas)), all_rmses[:,k], '-o', label=label)

plt.xticks(np.arange(len(alphas)), alphas)
plt.ylabel('Error')
plt.xlabel('alpha')
#plt.legend(loc=2)

ax.text(6, .21, '$\lambda = 0.1$', fontsize=12, color='blue')
ax.text(4, .175, '$\lambda = 0.3$', fontsize=12, color='green')
ax.text(8, .15, '$\lambda = 0.5$', fontsize=12, color='red')
ax.text(11, .22, '$\lambda = 0.7$', fontsize=12, color='darkcyan')
ax.text(11, .52, '$\lambda = 1$', fontsize=12, color='purple')
plt.show()
# world = random_walk(N_STATES)
# episode = world.generateEpisode()
# print(episode)
# print(len(episode))
#
# x,y = convertEpisodeToStateRep(episode)
# print(td_update(x, y, 1, 0.1))