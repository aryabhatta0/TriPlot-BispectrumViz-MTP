import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rotate_vector(vector, angle, axis):
    """
    Rotate a vector around a given axis by a specified angle.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    rotation_matrix = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (a * c + b * d)],
                                [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                                [2 * (b * d - a * c), 2 * (a * b + c * d), a * a + d * d - b * b - c * c]])
    return np.dot(rotation_matrix, vector)

def visualize_triangle(k1, k2, k3, z_cap, alpha, beta, gamma):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Rotate vectors k1, k2, k3 based on angles alpha, beta, gamma
    k1_rot = rotate_vector(k1, alpha, z_cap)
    k2_rot = rotate_vector(k2, beta, z_cap)
    k3_rot = rotate_vector(k3, gamma, z_cap)

    # Plot the vectors
    origin = np.zeros(3)
    ax.quiver(*origin, *k1_rot, color='r', label='k1')
    ax.quiver(*origin, *k2_rot, color='g', label='k2')
    ax.quiver(*origin, *k3_rot, color='b', label='k3')

    # Plot the triangle
    triangle_verts = [k1_rot, k2_rot, k3_rot]
    triangle_verts.append(triangle_verts[0])  # Repeat the first point to create a closed polygon
    triangle_xs, triangle_ys, triangle_zs = zip(*triangle_verts)
    ax.plot(triangle_xs, triangle_ys, triangle_zs, color='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangle Orientation')
    ax.legend()

    plt.show()

# Define vectors and angles
k1 = np.array([1, 0, 0])
k2 = np.array([0, 1, 0])
k3 = np.array([0, 0, 1])
z_cap = np.array([0, 0, 1])
alpha = np.pi/4
beta = np.pi/4
gamma = np.pi/4

# Visualize the triangle
visualize_triangle(k1, k2, k3, z_cap, alpha, beta, gamma)
