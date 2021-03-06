# Simple Machine Learned Interatomic Potentials (ML-IAPs)

Here we have Python notebooks explaining some very important concepts in neural network regression, and applications to potentials.

### `nn_quadratic_d1.ipynb`

Example of neural network regression where we teach a neural network to learn the shape of a parabola. Very similar to what happens in potential fitting, where the potential often has a somewhat parabolic shape. In this example, we build the neural network using matrix operations.

### `nn_quadratic_d2.ipynb`

Same as previous example, except instead of matrix operations we use built-in PyTorch modules like `nn.Linear` to build the network.

### `1d_chain.ipynb`

Here we make a 1D chain of atoms modeled by a simple harmonic potential `U=r^2`. We describe the atomic environment with Behler-Parinello descriptors, which are inputs to the neural network. Then we train a neural network potential to reproduce the target potential energies `U=r^2`. Next step is to add force fitting, which involves fitting the derivatives of the neural network.

### `1d_chain_d6.ipynb`

Same as previous, except cleaned up and we fit the neural network potential to forces here. The force fitting is slow because of loops in PyTorch... Need to think about how to speed this up!

### `1d_chain_d7.ipynb`

Here we verify that the neural network model forces are equivalent to those calculated via finite difference.
**Notes on fitting**:
- Good agreement for pure energy fitting: loss~1e-5
- This can take 3000-4000 epochs.
- Good agreement for energy + force fitting: loss~1e-4
- This can also take 3000-4000 epochs.
- Training time with forces is very slow - need to replace Python loops in force calculation with tensor operations.

### `nn_force_check_d4.py`

Compare NN model forces to finite difference forces. Model forces are calculated via the spatial gradient of the network energy, using a modified `compute_snap.cpp` located at https://github.com/rohskopf/lammps/tree/compute-snap.
