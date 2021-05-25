# Mixed Algorithms

## Conjugate Gradient

The conjugate gradient takes a function f which calculates a matrix vector product Ax for a 
vector x and the conjugate gradient algorithm then gives a solution to the equation system 
Ax=b , e.g. A^{-1}b.

## Vanilla Gradient 

Add vanilla gradients to your class. Simply implement a training objective, which can be 
differentiated by the model's parameters.

## Trust Region Gradient

Add trust region optimization to any class by inheriting from this one. For using trust regions
it is important that one gives a training distance, e.g. the KL or simply the L2-Norm and a training
objective. 

## Non Parametric Trust Region Optimization

When using the I-projection, it is interesting that for a sample based distribution, the 
result can be calculated in closed form, and this can be achieved by utilizing tbe non parametric
trust region methods.