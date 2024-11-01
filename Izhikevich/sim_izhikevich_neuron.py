import izhikevich as iz
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters
duration = 500
stepsize = 0.001

v=-65

# Regular Spiking
a=0.02
b=0.2
c=-65
d=8

# Chattering
a=0.02
b=0.2
c=-50
d=2

# Fast spiking
a=0.1
b=0.2
c=-65
d=2


time = np.arange(0.0,duration,stepsize)

n = iz.IzhikevichNeuron(v,a,b,c,d)

output = np.zeros(len(time))

# Run simulation
step = 0
i = 0
for t in time:
    if t>50:
        i = 5
    n.step(stepsize,i)
    output[step] = n.Voltage
    step += 1

# Plot activity
plt.plot(time,output)
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Neural activity")
plt.show()
