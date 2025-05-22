import numpy as np
import matplotlib.pyplot as plt

# Define constants for the power spectrum
A = 1.0  # Normalization constant for P(k)
f_nl = 10  # Strength of non-Gaussianity

# Define the power spectrum function
def power_spectrum(k):
    """Returns the power spectrum P(k) = A * k^-3"""
    return A * k**-3

# Define bispectrum templates
def bispectrum_local(k1, k2, k3, f_nl):
    """Local template for the bispectrum."""
    return 2 * f_nl * (power_spectrum(k1) * power_spectrum(k2) + 
                       power_spectrum(k2) * power_spectrum(k3) + 
                       power_spectrum(k3) * power_spectrum(k1))

def bispectrum_equilateral(k1, k2, k3, f_nl):
    """Corrected Equilateral template for the bispectrum."""
    P1, P2, P3 = power_spectrum(k1), power_spectrum(k2), power_spectrum(k3)
    
    # First term: -(P1 * P2 + P2 * P3 + P3 * P1)
    term1 = -(P1 * P2 + P2 * P3 + P3 * P1)
    
    # Second term: -2 * (P1 * P2 * P3)^(2/3)
    term2 = -2 * (P1 * P2 * P3)**(2/3)
    
    # Third term: P1^(1/3) * P2^(2/3) * P3 + cyc.
    term3 = P1**(1/3) * P2**(2/3) * P3 + P2**(1/3) * P3**(2/3) * P1 + P3**(1/3) * P1**(2/3) * P2
    
    # Combine terms and multiply by 6f_NL
    return 6 * f_nl * (term1 + term2 + term3)


# Helper functions
def calculate_k3(k1, k2, mu):
    """Calculates k3 using the triangle relation."""
    return np.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * mu)

# Parameters
mu = 0.5  # Fixed cos(theta)
t = 1.0  # Fixed k2/k1 ratio
k1_vals = np.linspace(0.01, 1, 100)  # Varying k1 values
k2_vals = t * k1_vals  # k2 is proportional to k1
k3_vals = calculate_k3(k1_vals, k2_vals, mu)  # Calculate k3 based on mu

# Calculate bispectrum for varying k1
bispectrum_local_vals = [bispectrum_local(k1, k2, k3, f_nl) 
                         for k1, k2, k3 in zip(k1_vals, k2_vals, k3_vals)]
bispectrum_equilateral_vals = [bispectrum_equilateral(k1, k2, k3, f_nl) 
                               for k1, k2, k3 in zip(k1_vals, k2_vals, k3_vals)]

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(k1_vals, bispectrum_local_vals, label='Local PNG', color='blue')
# plt.plot(k1_vals, bispectrum_equilateral_vals, label='Equilateral PNG', color='red')
plt.xlabel('$k_1$ (Scale)', fontsize=14)
plt.ylabel('Bispectrum $B_\Phi(k_1, k_2, k_3)$', fontsize=14)
plt.title('Scale Dependence of Bispectrum for Local PNG', fontsize=8)
plt.legend()
plt.grid()
plt.show()
