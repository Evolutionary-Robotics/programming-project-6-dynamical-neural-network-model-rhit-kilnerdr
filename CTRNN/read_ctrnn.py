import ctrnn
import matplotlib.pyplot as plt
import numpy as np

size = 50
duration = 50
stepsize = 0.01

time = np.arange(0.0,duration,stepsize)

nn = ctrnn.CTRNN(size)

nn.load("C:/Users/Drew/Desktop/RoseWork/Evolutionary-Robotics/P6/CTRNN/ctrnn4.npz")

nn.initializeState(np.zeros(size))

outputs = np.zeros((len(time),size))
states = np.zeros((len(time),size))
positions = np.zeros((len(time), 2))  # x, y position of the body

# Initial position and angle of the robot
position = np.array([0.0, 0.0])
angle = 0.0
speed_gain = 0.5  # scaling factor for speed based on neural output

# Run simulation
step = 0
for t in time:
    # Step the neural network
    nn.step(stepsize)
    outputs[step] = nn.Outputs
    states[step] = nn.States

    # Map outputs to wheel speeds
    left_wheel_speed = nn.Outputs[0] * speed_gain
    right_wheel_speed = nn.Outputs[1] * speed_gain

    # Calculate body movement
    linear_velocity = (left_wheel_speed + right_wheel_speed) / 2.0
    angular_velocity = (right_wheel_speed - left_wheel_speed) / 0.5  # assuming distance between wheels is 0.5 units

    # Update position and orientation
    angle += angular_velocity * stepsize
    position += linear_velocity * np.array([np.cos(angle), np.sin(angle)]) * stepsize

    # Save position for plotting
    positions[step] = position
    step += 1

# How much is the neural activity changing over time
activity = np.sum(np.abs(np.diff(outputs,axis=0)))/(duration*size*stepsize)
print("Overall activity: ",activity)

# Plot activity
plt.plot(time,outputs)
plt.xlabel("Time")
plt.ylabel("Outputs")
plt.title("Neural output activity")
plt.show()

# Plot activity
plt.plot(time,states)
plt.xlabel("Time")
plt.ylabel("States")
plt.title("Neural state activity")
plt.show()

# Plot robot path
plt.plot(positions[:, 0], positions[:, 1], 'r-', label="Path")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Robot Path")
plt.legend()
plt.axis("equal")
plt.show()

