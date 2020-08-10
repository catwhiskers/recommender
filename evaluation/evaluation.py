

def diversity(rcmds, target, sim_func):
    objs = rcmds + [target]
    scores = []
    for i, o in enumerate(objs):
        for j in range(0, i):
            scores.append(sim_func(o, objs[j]))
    return 1 - sum(scores) / len(scores)

def novelty(rcmds, p13n_history):
    return (len(rcmds) - len(rcmds.intersection(p13n_history)))/len(rcmds)




rcmds = [[1, 1, 1], [1, 0.5, 0], [2, 1, 0.8]]
target = [1, 1, 1]
import math
def sim_func(x, y):
    score = 0
    lenx = 0
    leny = 0
    for i in range(0, len(x)):
        score += x[i]*y[i]
        lenx += x[i]*x[i]
        leny += y[i]*y[i]
    return score / math.sqrt(lenx)/ math.sqrt(leny)
res = diversity(rcmds, target, sim_func)
print(res)

rcmd_items = set([1,2,3,4])
p13n_history = set([3,4])
res = novelty(rcmd_items, p13n_history)
print(res)



