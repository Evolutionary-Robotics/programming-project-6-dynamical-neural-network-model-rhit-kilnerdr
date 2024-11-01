import izhikevich as iz
import matplotlib.pyplot as plt
import numpy as np

# Global Parameters
size = 100
duration = 1000 #1000
stepsize = 0.01

time = np.arange(0.0,duration,stepsize)

nn = iz.IzhikevichNetwork(size)
nn.pcnnParams(8)

outputs = np.zeros((len(time),size))
firings = np.zeros((len(time),size))
positions = np.zeros((len(time), 2))  # x, y position of the body

# Initial position and angle of the robot
position = np.array([0.0, 0.0])
angle = 0.0
speed_gain = 0.5

# Run simulation
step = 0
inputs = np.zeros(size)
for t in time:
    if t > 10:
        inputs = 10 * np.ones(size)  # constant input after t > 10
    nn.step(stepsize, inputs)

    # Record neuron activity
    outputs[step] = nn.Voltages
    firings[step] = nn.Firing

    # Map neuron voltages to wheel speeds
    left_wheel_speed = nn.Voltages[0] * speed_gain
    right_wheel_speed = nn.Voltages[1] * speed_gain

    # Calculate linear and angular velocities
    linear_velocity = (left_wheel_speed + right_wheel_speed) / 2.0
    angular_velocity = (right_wheel_speed - left_wheel_speed) / 0.5

    # Update position and orientation
    angle += angular_velocity * stepsize
    position += linear_velocity * np.array([np.cos(angle), np.sin(angle)]) * stepsize

    # Store position for visualization
    positions[step] = position
    step += 1

# Plot activity
plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Neural activity")
plt.show()

# Plot activity
plt.imshow(firings.T, cmap='Greys', interpolation='nearest', aspect='auto')
plt.xlabel("Time")
plt.ylabel("Voltage")
plt.title("Neural activity")
plt.show()

# Plot robot path
plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Path")
# plt.scatter(positions[:, 0], positions[:, 1], s=10, c='blue', alpha=0.6, label="Positions")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Robot Path")
plt.legend()
plt.axis("equal")
plt.show()
