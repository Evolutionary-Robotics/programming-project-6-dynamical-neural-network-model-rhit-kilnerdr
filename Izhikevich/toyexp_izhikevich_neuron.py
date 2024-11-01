import matplotlib.pyplot as plt
import numpy as np

class IzhikevichNeuron:

    def __init__(self, v, a, b, c, d):
        self.Voltage = v
        self.u = b*v
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.spikecount = 0
        self.intervaldict = {}
        self.intervaltimecounter = 0.0

    def step(self,dt,Input):
        dVdt = (0.04 * (self.Voltage**2)) + (5 * self.Voltage) + 140.0 - self.u + Input
        dudt = self.a * ((self.b * self.Voltage) - self.u)
        self.Voltage += dt * dVdt
        self.u += dt * dudt
        if self.Voltage >= 30.0:
            self.Voltage = self.c
            self.u = self.u + self.d
            self.spikecount += 1
            if self.intervaltimecounter in self.intervaldict:
                self.intervaldict[self.intervaltimecounter] += 1
            else:
                self.intervaldict[self.intervaltimecounter] = 1

            self.intervaltimecounter = 0.0
        else:
            self.intervaltimecounter += dt

# Global Parameters
duration = 1000
stepsize = 0.001

v=-65

# Regular Spiking
a=0.02
b=0.2
c=-65
d=8

# # Chattering
# a=0.02
# b=0.2
# c=-50
# d=2
#
# # Fast spiking
# a=0.1
# b=0.2
# c=-65
# d=2

time = np.arange(0.0,duration,stepsize)
inputs = np.arange(0.0,20.0,0.5)
count = np.zeros(len(inputs))
k = 0
for inp in inputs:
    # Create neuron
    n = IzhikevichNeuron(v,a,b,c,d)
    # Run simulation
    i = 0
    for t in time:
        if t>50:
            i = inp
        n.step(stepsize,i)
    count[k] = n.spikecount
    k += 1
    for key in n.intervaldict:
        plt.plot(inp,key,'ko')

plt.xlabel("Input")
plt.ylabel("Interval")
plt.show()
