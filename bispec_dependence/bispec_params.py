import numpy as np
import matplotlib.pyplot as plt

# Define constants for the power spectrum
A = 1.0  # Normalization constant for P(k)
k_min = 0.01
k_max = 1.0
k_points = 100

# Define the power spectrum function
def power_spectrum(k):
    """Returns the power spectrum P(k) = A * k^-3"""
    return A * k**-3

def bispectrum_local(k1, k2, k3, f_nl):
    """Local template for the bispectrum."""
    return 2 * f_nl * (power_spectrum(k1) * power_spectrum(k2) + 
                       power_spectrum(k2) * power_spectrum(k3) + 
                       power_spectrum(k3) * power_spectrum(k1))

def bispectrum_equilateral(k1, k2, k3, f_nl):
    """Corrected Equilateral template for the bispectrum."""
    P1, P2, P3 = power_spectrum(k1), power_spectrum(k2), power_spectrum(k3)    
    term1 = -(P1 * P2 + P2 * P3 + P3 * P1)
    term2 = -2 * (P1 * P2 * P3)**(2/3)
    term3 = P1**(1/3) * P2**(2/3) * P3 + P2**(1/3) * P3**(2/3) * P1 + P3**(1/3) * P1**(2/3) * P2
    return 6 * f_nl * (term1 + term2 + term3)

def bispectrum_orthogonal(k1, k2, k3, f_nl):
    """Orthogonal template for the bispectrum."""
    P1, P2, P3 = power_spectrum(k1), power_spectrum(k2), power_spectrum(k3)
    term1 = -3 * (P1 * P2 + P2 * P3 + P3 * P1)
    term2 = -8 * (P1 * P2 * P3)**(2/3)
    term3 = 3 * (P1**(1/3) * P2**(2/3) * P3 +
                 P2**(1/3) * P3**(2/3) * P1 +
                 P3**(1/3) * P1**(2/3) * P2)
    return 6 * f_nl * (term1 + term2 + term3)

def bispectrum_folded(k1, k2, k3, f_nl=10):
    """Folded template for the bispectrum."""
    P1, P2, P3 = power_spectrum(k1), power_spectrum(k2), power_spectrum(k3)
    term1 = (P1 * P2 + P2 * P3 + P3 * P1)
    term2 = 3 * (P1 * P2 * P3)**(2/3)
    term3 = - (P1**(1/3) * P2**(2/3) * P3 +
                 P2**(1/3) * P3**(2/3) * P1 +
                 P3**(1/3) * P1**(2/3) * P2)
    return 6 * f_nl * (term1 + term2 + term3)

# Reduced bispectrum Q
def reduced_bispectrum(bispectrum, k1, k2, k3):
    """Calculates the reduced bispectrum Q."""
    denominator = power_spectrum(k1) * power_spectrum(k2) + power_spectrum(k2) * power_spectrum(k3) + power_spectrum(k3) * power_spectrum(k1)
    return bispectrum / denominator

# Angular dependence helper
def triangle_side(k1, k2, theta):
    """Returns the third side of a triangle given two sides and the angle between them."""
    return np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * np.cos(theta))

# Visualization of Bispectra
def plot_bispectrum(template, f_nl=10, k_fixed=0.1):
    """Plots the bispectrum for a fixed k1 value and varying k2, k3 configurations."""
    k2_vals = np.linspace(k_min, k_max, k_points)
    theta_vals = np.linspace(0, np.pi, k_points)  # Angle between k1 and k2
    k3_vals = [triangle_side(k_fixed, k2, theta) for k2 in k2_vals for theta in theta_vals]

    bispectrum_vals = []
    for k2, k3 in zip(k2_vals, k3_vals):
        bispectrum_vals.append(template(k_fixed, k2, k3, f_nl))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(k2_vals, bispectrum_vals, label=f'f_nl = {f_nl}')
    plt.xlabel('$k_2$', fontsize=14)
    plt.ylabel('$B_\Phi(k_1, k_2, k_3)$', fontsize=14)
    plt.title(f'Bispectrum for {template.__name__} template', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()

def plot_bispectrum_fixed_theta(template, f_nl=10, k_fixed=0.1, theta=np.pi/3):
    """Plots the bispectrum for a fixed k1 and theta, varying k2."""
    k2_vals = np.linspace(k_min, k_max, k_points)
    k3_vals = [triangle_side(k_fixed, k2, theta) for k2 in k2_vals]  # Fixed theta
    
    bispectrum_vals = []
    for k2, k3 in zip(k2_vals, k3_vals):
        bispectrum_vals.append(template(k_fixed, k2, k3, f_nl))
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(k2_vals, bispectrum_vals, label=f'f_nl = {f_nl}, θ = {theta:.2f} rad')
    plt.xlabel('$k_2$', fontsize=14)
    plt.ylabel('$B_\Phi(k_1, k_2, k_3)$', fontsize=14)
    plt.title(f'{template.__name__} PNG', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()


# Example: Generate plots for each template
plot_bispectrum_fixed_theta(bispectrum_local, f_nl=10, k_fixed=0.1)
plot_bispectrum_fixed_theta(bispectrum_equilateral, f_nl=10, k_fixed=0.1)
plot_bispectrum_fixed_theta(bispectrum_orthogonal, f_nl=10, k_fixed=0.1)
plot_bispectrum_fixed_theta(bispectrum_folded, f_nl=10, k_fixed=0.1)

# Angular Dependence Visualization
def plot_angular_dependence(template, k1=0.1, k2=0.2, f_nl=10):
    """Plots the bispectrum as a function of angle theta between k1 and k2."""
    theta_vals = np.linspace(0, np.pi, k_points)
    k3_vals = [triangle_side(k1, k2, theta) for theta in theta_vals]

    bispectrum_vals = []
    for k3, theta in zip(k3_vals, theta_vals):
        bispectrum_vals.append(template(k1, k2, k3, f_nl))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(theta_vals, bispectrum_vals, label=f'f_nl = {f_nl}')
    plt.xlabel(r'$\theta$ (radians)', fontsize=14)
    plt.ylabel('$B_\Phi(k_1, k_2, k_3)$', fontsize=14)
    plt.title(f'Angular Dependence of Bispectrum for {template.__name__} template', fontsize=16)
    plt.legend()
    plt.grid()
    plt.show()

# Example: Plot angular dependence for each template
plot_angular_dependence(bispectrum_equilateral, k1=0.1, k2=0.2, f_nl=10)
plot_angular_dependence(bispectrum_orthogonal, k1=0.1, k2=0.2, f_nl=10)
plot_angular_dependence(bispectrum_folded, k1=0.1, k2=0.2, f_nl=10)
