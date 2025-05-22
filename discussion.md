**Theoretical Background:**
- t= k2/k1 and μ = cosθ
- P(k) = A* k**(-3)

**Discussions:**
- Plot the ratio of bispectrum to cyclic combination of power spectrum. It will be dimensionless B / P(k1)*P(k2) + cycl
- for each type of non-Gaussianity for some k1 value
- Study how bispectrum depends on shape of triangle
- What is dimension of bispectrum - dimensionless

- convert Bphi to matter bispectrum
- A can be calculated from sigma8 value
- P = k^(n_s-4)
- matter power spectrum in order to calculate A
- look into scale of k1
- show the scale: k1 => lambda1 => showing in a grid
- lambda=2 => lines at 2 dist

**Visualization:**
- logarithmic scale -> fixing color bars
- how can we differentiate between equilateral and orthogonal bispectrum
- this is fourier space; grid would be in real space interesting to see

Power spectrum visualization is easy, bispectrum is hard.

## TASKS

* there are several types of bispectrum- local, equilateral, orthogonal: the bispectrum formula for each is given in paper
* The bispectrums are parametrized using P1,P2,P3 power spectrums which is also dependent on k1,k2,k3 wave vectors.
* Remember the mu-t space of bispectrum? mu=cos(theta), t = k2/k1. k1 is fixed, and size of triangle is varied using mu and t. I've also attached the paper with details about this.
* So any (mu, t) will give a (k1,k2,k3) which will give a (P1,P2,P3) giving a bispectrum value.

We want to make a visualizing software in Python which will have:
* an option to choose which type of bispectrum we want to visualize
* A mu-t input space where user can decide mu-t value by cursor movements in the allowed region.
* A corresponding k1-k2-k3 triangle can be shown optionally.
* We'll have a P1, P2, P3 value.
* We can calculate: res = Bispec / (P1P2 + P2P3 + P3P1) , which will be a unitless quantity
* `res is what we want to visualize as cursor moves in mu-t input space.`
