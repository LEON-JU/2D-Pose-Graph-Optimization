import numpy as np
import matplotlib.pyplot as plt

'''
The original_poses and optimized_poses are both copied from the output of ./trajectory_optimization
'''

original_poses = [
    0, 0, 0,
    -0.146303, 0.52145, 0.463171,
    -0.483216, 0.881682, 1.019,
    -0.956247, 0.992529, 1.47808,
    -1.5145, 0.919746, 1.97284,
    -1.96216, 0.599956, 2.55723,
    -2.14286, 0.136549, 3.13284,
    -2.00857, -0.330487, -2.64456,
    -1.63789, -0.746245, -2.07607,
    -1.11791, -0.885967, -1.5782,
    -0.632563, -0.731991, -1.11388,
    -0.237955, -0.40841, -0.64978
]

optimized_poses = [
    6.89658e-17, -1.96267e-16, 2.98358e-16,
    -0.142843, 0.510642, 0.486148,
    -0.484483, 0.85223, 1.06245,
    -0.958422, 0.931622, 1.53715,
    -1.50794, 0.815206, 2.04118,
    -1.92926, 0.454785, 2.62905,
    -2.07279, -0.0312017, 3.2049,
    -1.90176, -0.498165, -2.57202,
    -1.49846, -0.896772, -2.00008,
    -0.965912, -1.00742, -1.49338,
    -0.491896, -0.82369, -1.01447,
    -0.12789, -0.473351, -0.530638
]

original_poses = np.array(original_poses).reshape(-1, 3)
optimized_poses = np.array(optimized_poses).reshape(-1, 3)

# Generate ground truth circular trajectory
angles = np.linspace(0, 2 * np.pi, num=len(original_poses), endpoint=False)
radius = 1
ground_truth_x = radius * np.cos(angles) - 1
ground_truth_y = radius * np.sin(angles)

# visualization
plt.figure()
plt.plot(original_poses[:, 0], original_poses[:, 1], '-o', label='Original Poses')
plt.plot(ground_truth_x, ground_truth_y, '-o', label='Ground Truth')
plt.plot(optimized_poses[:, 0], optimized_poses[:, 1], '-o', label='Optimized Poses')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectories visualization')
plt.grid(True)
plt.legend()
plt.axis('equal')

plt.show()
