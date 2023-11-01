import numpy as np
import pickle
#### Script to load the data from the simulation
#with open('changing5.pkl', 'rb') as f:
#    x_tilde_lists = pickle.load(f)
values=np.load("tracking20.npz")
tracking=values['tracking_error']
tracking_rel=values['tracking_rel']
#array_reinforce=values['reinforce']
print("Hi")
