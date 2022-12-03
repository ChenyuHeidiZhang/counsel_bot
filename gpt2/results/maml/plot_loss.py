filename = 'support=1.model=small.mode=last/inner_steps:4.inner_lr:0.001.outer_lr:0.0005.txt'

iter = []
losses = []
with open(filename, 'r') as f:
    for line in f:
        if line.startswith('Validation'): continue
        splitted = line.split(': loss: ')
        losses.append(float(splitted[1]))
        iter.append(int(splitted[0].split('Iteration ')[-1]))

# print(losses)
# print(iter)

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.plot(iter, losses)
plt.savefig('plot_maml.png')