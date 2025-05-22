import numpy as np
import matplotlib.pyplot as plt
from scipy.special import binom
from typing import Tuple, Optional

def power_spectrum(k: float, A_s: float = 2.1e-9, n_s: float = 0.965, k_pivot: float = 0.05) -> float:
    """
    Linear matter power spectrum.
    
    Args:
        k: Wave number magnitude
        A_s: Amplitude of scalar fluctuations
        n_s: Spectral index
        k_pivot: Pivot scale
        
    Returns:
        Power spectrum value P(k)
    """
    return A_s * (k / k_pivot)**(n_s - 1) * k**(-3)


def bispectrum_gaussian(k1: float, k2: float, k3: float, f_nl: Optional[float] = None) -> float:
    """
    Gaussian bispectrum (trivial case).
    
    Args:
        k1, k2, k3: Wave number magnitudes
        f_nl: Non-Gaussianity parameter (unused for Gaussian case)
        
    Returns:
        Bispectrum value (always 0 for Gaussian - no three-point correlations)
    """
    return 0.0  # Gaussian case has no bispectrum signal


def bispectrum_local(k1: float, k2: float, k3: float, f_nl: float) -> float:
    """
    Local-type non-Gaussian bispectrum according to thesis Eq. 2.1.
    B^local(k1,k2,k3) = 2*f_NL * [P_φ(k1)*P_φ(k2) + cyc.]
    
    Args:
        k1, k2, k3: Wave number magnitudes
        f_nl: Non-Gaussianity parameter
        
    Returns:
        Local bispectrum value
    """
    P1 = power_spectrum(k1)
    P2 = power_spectrum(k2)
    P3 = power_spectrum(k3)
    
    # Local bispectrum: 2*f_NL * [P(k1)*P(k2) + P(k2)*P(k3) + P(k3)*P(k1)]
    return 2 * f_nl * (P1 * P2 + P2 * P3 + P3 * P1)


def bispectrum_equilateral(k1: float, k2: float, k3: float, f_nl: float) -> float:
    """
    Equilateral-type non-Gaussian bispectrum according to thesis Eq. 2.2.
    
    Args:
        k1, k2, k3: Wave number magnitudes
        f_nl: Non-Gaussianity parameter
        
    Returns:
        Equilateral bispectrum value
    """
    P1 = power_spectrum(k1)
    P2 = power_spectrum(k2)
    P3 = power_spectrum(k3)
    
    # Equilateral bispectrum from thesis Eq. 2.2
    term1 = -(P1 * P2 + P2 * P3 + P3 * P1)
    term2 = -2 * (P1 * P2 * P3)**(2/3)
    term3 = (P1**(1/3) * P2**(2/3) * P3 + P2**(1/3) * P3**(2/3) * P1 + P3**(1/3) * P1**(2/3) * P2)
    
    return (6 * f_nl) * (term1 + term2 + term3)

def is_valid_triangle(k1: float, k2: float, k3: float, tolerance: float = 1e-10) -> bool:
    """
    Check if three wave numbers form a valid triangle configuration.
    """
    return (k1 + k2 > k3 + tolerance and 
            k1 + k3 > k2 + tolerance and 
            k2 + k3 > k1 + tolerance)


def plot_bispectrum_kspace(bispectrum_func, f_nl: float, kmin: float = 0.1, 
                          kmax: float = 1.0, nk: int = 100, title: str = "Bispectrum"):
    """
    Plot bispectrum in k-space (2D slice at fixed k3).
    
    Args:
        bispectrum_func: Function to compute bispectrum values
        f_nl: Non-Gaussianity parameter
        kmin, kmax: Range of k values
        nk: Number of grid points
        title: Plot title
    """
    k_vals = np.linspace(kmin, kmax, nk)
    k3_fixed = k_vals[nk // 2]  # Fix k3 at middle value
    
    # Create meshgrid for k1 and k2
    K1, K2 = np.meshgrid(k_vals, k_vals)
    bispectrum = np.zeros_like(K1)
    
    # Calculate bispectrum values
    for i in range(nk):
        for j in range(nk):
            k1, k2 = K1[i, j], K2[i, j]
            if is_valid_triangle(k1, k2, k3_fixed):
                bispectrum[i, j] = bispectrum_func(k1, k2, k3_fixed, f_nl)
            else:
                bispectrum[i, j] = np.nan  # Invalid configurations
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(bispectrum, extent=[kmin, kmax, kmin, kmax], 
                   origin='lower', cmap='viridis', aspect='auto')
    ax.set_xlabel(r'$k_1$', fontsize=14)
    ax.set_ylabel(r'$k_2$', fontsize=14)
    ax.set_title(f'{title} in k-space (k₃={k3_fixed:.2f})', fontsize=16)
    plt.colorbar(im, ax=ax, label='Bispectrum')
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run bispectrum analysis and plotting."""
    # Parameters
    f_nl = 1.0
    kmin = 0.1
    kmax = 1.0
    nk = 100
    samples = 15000
    
    print("=== Bispectrum Analysis ===\n")
    
    # Define bispectrum models
    models = [
        (bispectrum_gaussian, "Gaussian"),
        (bispectrum_local, "Local"),
        (bispectrum_equilateral, "Equilateral")
    ]
    
    # Plot in k-space
    print("Plotting bispectra in k-space...")
    for func, name in models:
        print(f"  - {name} model")
        plot_bispectrum_kspace(func, f_nl, kmin, kmax, nk, name)


if __name__ == "__main__":
    main()