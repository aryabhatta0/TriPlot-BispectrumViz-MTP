import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to rotate a vector around an axis
def rotate_vector(vector, axis, angle):
    rotation_matrix = (
        np.cos(angle) * np.eye(3) +
        (1 - np.cos(angle)) * np.outer(axis, axis) +
        np.sin(angle) * np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
    )
    return np.dot(rotation_matrix, vector)

# Define triangle_plot as a global variable
triangle_plot = None

# Function to update the triangle plot
def update_triangle_plot(pz, px):
    global triangle_plot  # Declare triangle_plot as a global variable
    # Calculate py
    py = np.sqrt(1 - pz**2 - px**2)
    
    # Shape and size fixed
    mod_k1 = 1
    mu = 0.8
    t = 0.8

    # Define k1, k2, and k3
    k1 = np.array([0, 0, mod_k1])
    k2 = np.array([t*mod_k1*np.sqrt(1-mu*mu), 0, -t*mod_k1*mu])
    k3 = -(k1 + k2)

    # Calculate mu1, mu2, and mu3
    mu1 = pz
    mu2 = -mu*pz + np.sqrt(1 - mu**2)*px
    mu3 = -((1-t*mu)*pz + t*np.sqrt(1-mu**2)*px) / np.sqrt(1- 2*t*mu + t**2)

    # Calculate angles
    alpha = np.arccos(mu1)
    beta = np.arccos(mu2)
    gamma = np.arccos(mu3)

    # Define line of sight vector z_cap
    z_cap = np.array([0, 0, 1])

    # Rotate vectors k1, k2, and k3
    k1_rotated = rotate_vector(k1, z_cap, alpha)
    k2_rotated = rotate_vector(k2, z_cap, beta)
    k3_rotated = rotate_vector(k3, z_cap, gamma)

    # Update the triangle plot
    if triangle_plot is None:
        # If triangle_plot is not yet defined, create it
        triangle_plot, = triangle_ax.plot([], [], [], color='gray', linestyle='--', label='Triangle')
    triangle_verts = [k1_rotated, k2_rotated, k3_rotated, k1_rotated]
    triangle_xs, triangle_ys, triangle_zs = zip(*triangle_verts)
    triangle_plot.set_data(triangle_xs, triangle_ys)
    triangle_plot.set_3d_properties(triangle_zs)

# Function to handle mouse movement event
def on_mouse_move(event):
    if event.inaxes == circle_ax:
        # Update pz and px based on cursor position
        pz = event.ydata
        px = event.xdata
        # Update the triangle plot
        update_triangle_plot(pz, px)
        # Redraw the plot
        plt.draw()

# Create the figure and axis for circle plot
circle_fig, circle_ax = plt.subplots()
circle_ax.set_aspect('equal')
circle_ax.set_xlim(-1, 1)
circle_ax.set_ylim(-1, 1)
circle_ax.set_xlabel('px')
circle_ax.set_ylabel('pz')
circle_ax.plot([0], [0], 'ro')  # Plot the origin

# Create the circle plot
circle = plt.Circle((0, 0), 1, fill=False)
circle_ax.add_artist(circle)

# Create the figure and axis for triangle plot
triangle_fig = plt.figure()
triangle_ax = triangle_fig.add_subplot(111, projection='3d')

# Initialize pz and px values
pz = 0.2
px = 0.2

# Update the triangle plot initially
update_triangle_plot(pz, px)

# Plot the triangle initially
triangle_plot, = triangle_ax.plot([], [], [], 'b-', label='Triangle')

# Set labels and legend for triangle plot
triangle_ax.set_xlabel('X')
triangle_ax.set_ylabel('Y')
triangle_ax.set_zlabel('Z')
triangle_ax.legend()

# Connect the mouse movement event to the mouse move handler function
circle_fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Show the plots
plt.show()
