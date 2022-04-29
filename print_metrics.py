import json
import numpy as np
from matplotlib import pyplot as plt

def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

losses = []
iters = []
experiment_metrics = load_json_arr('results/train roi with 6pbase/metrics.json')
for line in experiment_metrics:
    if 'loss_base_reg' in line and 'data_time' in line and line['data_time'] > 0.03:
        losses.append(line['loss_base_reg'])
        iters.append(line['iteration'])
        print(line['loss_base_reg'])

losses = np.array(losses)
iters = np.array(iters)
# plt.plot(iters[-2227:], losses[-2227:])
plt.plot(iters[-1000:], losses[-1000:])
plt.show()