import json 
import numpy as np 

data = []

for _ in range(10):
    for rho in np.arange(0.05, 1.001, 0.01):
        data.append({ "samples": 10000, "testsize": 0.0, "method": "drep", "div": round(rho,2) })

with open('_config_hyperparam_1.json', 'w') as json_file:
  json.dump(data, json_file)