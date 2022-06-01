import numpy as np
import copy
from scipy.spatial import distance_matrix

from matplotlib import pyplot as plt


robot_num = 5
task_num = 50

robot = np.random.uniform(0,1,(robot_num,2))
task = np.random.uniform(0,1,(task_num,2))

# Sequential Greedy Algorithm
I = [i for i in range(len(robot))]
J = [j for j in range(len(task))]

bundle = [[] for i in range(len(robot))]
path = [[] for i in range(len(robot))]
Sp = [0 for i in range(len(robot))]

L_t = len(task)
N_min = min(len(task),len(robot)*L_t)
alpha = 1
penalty = 100

eta = np.zeros(len(robot),dtype=np.int16)
score = -distance_matrix(robot,task)

for n in range(N_min):
    i_star,j_star = np.unravel_index(np.argmax(score, axis=None), score.shape)

    eta[i_star] += 1
    J.remove(j_star)

    # Construct Bundle
    bundle[i_star].append(j_star)

    # Construct Path
    if len(path[i_star]) > 0:
        c_list = []
        for k in range(len(path[i_star])+1):
            p = copy.deepcopy(path[i_star])
            p.insert(k,j_star)
            c = 0
            c += distance_matrix([robot[i_star]],[task[p[0]]]).squeeze()
            if len(p)>1:
              for loc in range(len(p)-1):
                c += distance_matrix([task[p[loc]]],[task[p[loc+1]]]).squeeze()
            c = -(c-Sp[i_star])
            c_list.append(c)

        idx = np.argmax(c_list)
        c_max = c_list[idx]

        path[i_star].insert(idx,j_star)
    else:
      path[i_star].append(j_star)

    Sp[i_star] = 0
    Sp[i_star] += distance_matrix([robot[i_star]],[task[path[i_star][0]]]).squeeze()
    if len(path[i_star])>1:
      for loc in range(len(path[i_star])-1):
          Sp[i_star] += distance_matrix([task[path[i_star][loc]]],[task[path[i_star][loc+1]]]).squeeze()

    max_length = max(Sp)

    if eta[i_star] == L_t:
        I.remove(i_star)
        score[i_star,:] = -np.inf

    score[I,j_star] = -np.inf

    # Score Update
    for i in I:
        for j in J:
            c_list = []
            for k in range(len(path[i])+1):
                p = copy.deepcopy(path[i])
                p.insert(k,j)
                c = 0
                c += distance_matrix([robot[i]],[task[p[0]]]).squeeze()
                if len(p)>1:
                  for loc in range(len(p)-1):
                    c += distance_matrix([task[p[loc]]],[task[p[loc+1]]]).squeeze()

                if c > max_length:
                  c = -(c-max_length)*penalty
                else:
                  c=-(c-Sp[i])
                c_list.append(c)

            if c_list:
              if i == i_star:
                  score[i,j] = (1/alpha) * max(c_list)
              else:
                  score[i,j] = max(c_list)

print("Path",bundle)
print("Bundle",path)
print("Max Route:",max(Sp))

fig, ax = plt.subplots()
ax.set_xlim((-0.1,1.2))
ax.set_ylim((-0.1,1.2))
ax.set_aspect("equal")

ax.set_title("Sequential Greedy Algorithm (MinMax)")
ax.plot(robot[:,0],robot[:,1],'r^',label="robot")
ax.plot(task[:,0],task[:,1],'bo',label="task")

for i in range(len(path)):
  p = copy.deepcopy(path[i])
  if p:
    ax.plot([robot[i,0],task[p[0],0]],[robot[i,1],task[p[0],1]],'r-')
    ax.plot(task[p,0],task[p,1],'r-')

ax.grid()
ax.legend(loc="upper right")

plt.show()
