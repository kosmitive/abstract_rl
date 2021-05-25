# Policy

There are two types of policies currently implemented.

# Continuous Policy

Two types are implemented the Gaussian and the Beta policy. Whereas we experienced numerical instabilities when
using the beta policy so it is not used in the current implementation. There are various methods implemented
up on these. For example the KL-divergence, Entropy and several other methods. Sometimes it makes 
sense that these methods are overwritten by special functions, e.g. when the KL-divergence can
be solved in a closed solution, as it is for Gaussians. The included EnvironmentWrapper accepts a policy
executes it and returns the obtained results.

# Non Parametric Policy 

This is used by the E-Step in MPO. It basically consists of samples and the Non Parametric Policy Builder.

