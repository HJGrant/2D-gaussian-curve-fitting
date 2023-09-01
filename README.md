# gaussian curve fitting for a laser dot.

This program comutes a optimally fitting curve for a sample of laser dots, and then uses that fit to find lasers in other image samples.

The two numpy arrays contained in the repo are a fit calculated from 25 laser samples, and the gaussian paramters that created that fit. The fit can be used for similarity calculations and the parameters can be passed to a 2D gaussian function for computing the fit. 
