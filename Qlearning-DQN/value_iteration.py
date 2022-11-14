import numpy as np


map = "S...H...H...GB"
states = np.arange(len(map))
q = np.zeros(len(states))
p = np.zeros((len(states), len(states)))
for s in states:
    if s == 4 or s == 8:
        p[s, s + 1] = 0.8
        p[s, -1] = 0.2
    elif s == 12 or s == 13:
        pass
    else:
        p[s, s + 1] = 1
r = np.zeros(len(states))
r[-2] = 10
r[-1] = -5

lam = 0.9
for i in range(100):
    q = r + lam * np.matmul(p, q)
    
