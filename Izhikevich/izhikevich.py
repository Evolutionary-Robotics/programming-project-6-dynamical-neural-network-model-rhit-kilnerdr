import numpy as np

class IzhikevichNeuron:

    def __init__(self, v, a, b, c, d):
        self.Voltage = v
        self.u = b*v
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def step(self,dt,Input):
        dVdt = (0.04 * (self.Voltage**2)) + (5 * self.Voltage) + 140.0 - self.u + Input
        dudt = self.a * ((self.b * self.Voltage) - self.u)
        self.Voltage += dt * dVdt
        self.u += dt * dudt
        if self.Voltage >= 30.0:
            self.Voltage = self.c
            self.u = self.u + self.d

class IzhikevichNetwork:

    def __init__(self, size):
        self.Size = size                        # number of neurons in the network
        self.Voltages = np.zeros(size) - 65     # neuron activation vector
        self.u = np.zeros(size)                 # neuron activation vector
        self.a = np.ones(size)                  # a
        self.b = np.zeros(size)                 # b
        self.c = np.zeros(size)                 # c
        self.d = np.zeros(size)                 # d
        self.Weights = np.zeros((size,size))    # weight matrix
        self.Inputs = np.zeros(size)            # neuron output vector
        self.Firing = np.zeros(size)

    def pcnnParams(self, numberExcitatory):
        for i in range(self.Size):
            if i < numberExcitatory:
                self.Weights[i] = 0.5*np.random.uniform(0,1,size=self.Size)
                r = np.random.random()
                self.a[i] = 0.02
                self.b[i] = 0.2
                self.c[i] = -65+15*r**2
                self.d[i] = 8-6*r**2
            else:
                self.Weights[i] = -0.5*np.random.uniform(0,1,size=self.Size)
                r = np.random.random()
                self.a[i] = 0.02+0.08*r
                self.b[i] = 0.25-0.05*r
                self.c[i] = -65
                self.d[i] = 2

    def step(self, dt, inputs):
        self.Inputs += inputs
        # Update state
        dVdt = (0.04 * (self.Voltages**2)) + (5 * self.Voltages) + 140.0 - self.u + self.Inputs
        dudt = self.a * ((self.b * self.Voltages) - self.u)
        self.Voltages += dt * dVdt
        self.u += dt * dudt
        # Detect firings
        self.Firing = np.where(self.Voltages < 30.0, 0, 1)
        self.Voltages = self.Voltages * (1-self.Firing) + self.c * self.Firing
        self.u = self.u + self.d * self.Firing
        # Prepare synaptic input for next step
        self.Inputs = np.dot(self.Weights, self.Firing) ### Check if it should be .T or not
        #print(self.Weights, firing, self.Inputs)
