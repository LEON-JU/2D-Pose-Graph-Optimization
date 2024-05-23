import numpy as np
import matplotlib.pyplot as plt

Traj = np.array([[-0.146303,	-0.140469,	-0.153574,	-0.124153,	-0.119121,	-0.104924,	-0.138374,	-0.127593,	-0.129433,	-0.157566,	-0.116292,	-0.137706],
        [0.521450,	0.472806,	0.460935,	0.549113,	0.537101,	0.486201,	0.465843,	0.542203,	0.522633,	0.484192,	0.496887,	0.480551],
        [0.463171,	0.555832,	0.459081,	0.494756,	0.584395,	0.575603,	0.505790,	0.568483,	0.497871,	0.464326,	0.464097,	0.507884]])

def get_pose(matrix, i):
    column = matrix[:, i]
    return column[0], column[1], column[2]

def get_xy(pose):
    return pose[0][2], pose[1][2]

def visualization(Poses):
    fig, ax = plt.subplots()

    original_x = []
    original_y = []
    
    for i, pose in enumerate(Poses):
        x, y = get_xy(pose)
        original_x.append(x)
        original_y.append(y)
    
    plt.plot(original_x, original_y, '-o', label='Dead reckoning poses')

    # plot ground truth
    angles = np.linspace(0, 2 * np.pi, num=12, endpoint=False)
    radius = 1
    ground_truth_x = radius * np.cos(angles) - 1
    ground_truth_y = radius * np.sin(angles)
    ground_truth_x = np.append(ground_truth_x, ground_truth_x[0])
    ground_truth_y = np.append(ground_truth_y, ground_truth_y[0])
    plt.plot(ground_truth_x, ground_truth_y, '-o', label='Ground Truth')

    ax.set_aspect('equal', 'box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Dead reckoning visualization')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    Poses = []
    Poses.append(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))   # initial pose

    # dead reckoning
    for i in range(12):
        delta_x ,delta_y, theta = get_pose(Traj, i)
        Transformation = np.array([[np.cos(theta), -np.sin(theta), delta_x],
                                [np.sin(theta), np.cos(theta), delta_y],
                                [0, 0, 1]])
        new_pose = np.dot(Poses[i], Transformation)
        Poses.append(new_pose)

    # visualization
    visualization(Poses)