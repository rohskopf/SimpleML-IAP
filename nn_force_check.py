from __future__ import print_function
import sys
import ctypes
from ctypes import c_double
import numpy as np
from lammps import lammps, LMP_TYPE_ARRAY, LMP_STYLE_GLOBAL
from matplotlib import pyplot as plt
import torch
from torch import nn
from torch import tensor

def create_torch_network(layer_sizes):
    """
    Creates a pytorch network architecture from layer sizes.
    This also performs standarization in the first linear layer.
    This only supports softplus as the nonlinear activation function.

        Parameters:
            layer_sizes (list of ints): Size of each network layers

        Return:
            Network Architecture of type neural network sequential

    """
    layers = []
    try:

        layers.append(torch.nn.Linear(layer_sizes[0], layer_sizes[0]))
        for i, layer in enumerate(layer_sizes):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.Softplus())
            #layers.append(torch.nn.ReLU())
    except IndexError:
        layers.pop()

    # Fill weights with ones
    """
    nlayers = len(layers)
    print(f"{nlayers} layers.")
    for l in range(0,nlayers):
        print(layers[l])
        if (isinstance(layers[l],nn.Linear)):
            print(f"Linear layer l={l}")
            layers[l].weight.data.fill_(1.0)
            layers[l].bias.data.fill_(0.05)
    """

    return torch.nn.Sequential(*layers)

"""
Define the model
"""
class FitTorch(torch.nn.Module):
    """
    FitSNAP PyTorch Neural Network Architecture Model
    Currently only fits on energies
    """

    def __init__(self, network_architecture, descriptor_count, n_elements=1):
        """
        Saves lammps ready pytorch model.

            Parameters:
                network_architecture : A nn.Sequential network architecture
                descriptor_count (int): Length of descriptors for an atom
                n_elements (int): Number of differentiable atoms types

        """
        super().__init__()
        self.network_architecture = network_architecture
        self.desc_len = descriptor_count
        self.n_elem = n_elements

    def forward(self, x, xd, indices, atoms_per_structure, force_indices):
        """
        Saves lammps ready pytorch model.

            Parameters:
                x (tensor of floats): Array of descriptors
                x_derivatives (tensor of floats): Array of descriptor derivatives
                indices (tensor of ints): Array of indices upon which to contract per atom energies
                atoms_per_structure (tensor of ints): Number of atoms per configuration

        """
        #print(x_derivatives.size())
        nbatch = int(x.size()[0]/natoms)
        #print(f"{nbatch} configs in this batch")

        # Calculate energies
        predicted_energy_total = torch.zeros(atoms_per_structure.size())
        predicted_energy_total.index_add_(0, indices, self.network_architecture(x).squeeze())


        # Calculate forces
        x_indices = force_indices[0::3]
        y_indices = force_indices[1::3]
        z_indices = force_indices[2::3]
        #print(np.shape(force_indices))
        atom_indices = torch.tensor(force_indices[0::3,1].astype(int)-1,dtype=torch.long) # Atoms i are repeated for each cartesian direction
        neigh_indices = torch.tensor(force_indices[0::3,0].astype(int)-1,dtype=torch.long) # Neighbors j are repeated for each cartesian direction
        #print(neigh_indices.size())
        #print(int(neigh_indices))
        #dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)))
        dEdD = torch.autograd.grad(self.network_architecture(x), x, grad_outputs=torch.ones_like(self.network_architecture(x)))
        #print(dEdD[0])
        dEdD = dEdD[0][neigh_indices,:] # These need to be dotted with dDdR in the x, y, and z directions.
        #print(dEdD)
        dDdRx = xd[0::3]
        #print(dDdRx)
        #print(x)
        #print(dEdD.size())
        #print(dDdRx.size())
        elementwise = torch.mul(dDdRx, dEdD)
        #print(elementwise)
        # Need to contract these along rows with indices given by force_indices[:,1]
        #print(atom_indices)
        fx_components = torch.zeros(natoms,nd)
        #print(fx_components.size())
        contracted = fx_components.index_add_(0,atom_indices,elementwise)
        #print(contracted.size())
        # Sum along bispectrum components to get force on each atom.
        predicted_forces = torch.sum(contracted, dim=1)
        #print(predicted_forces.size())
        #print(x)
        #print(dEdD)
        #predicted_forces = torch.zeros(nconfigs*natoms)
        """
        # Loop over all configs given by number of rows in descriptors array
        for m in range(0,nbatch):
            for i in range(0,natoms):
                # Loop over neighbors of i
                numneighs_i = len(neighlists[m,i])
                for jj in range(0,numneighs_i):
                    j = neighlists[m,i,jj]
                    jtag = tags[m,j]
                    for k in range(0,nd):
                        predicted_forces[natoms*m + i] -= x_derivatives[natoms*m + i,(jj*nd)+k]*dEdD[0][natoms*m + jtag,k]
        """

        return (predicted_energy_total, predicted_forces)
        #return predicted_energy_total

    def import_wb(self, weights, bias):
        """
        Imports weights and bias into FitTorch model

            Parameters:
                weights (list of numpy array of floats): Network weights at each layer
                bias (list of numpy array of floats): Network bias at each layer

        """

        assert len(weights) == len(bias)
        imported_parameter_count = sum(w.size + b.size for w, b in zip(weights, bias))
        combined = [None] * (len(weights) + len(bias))
        combined[::2] = weights
        combined[1::2] = bias

        assert len([p for p in self.network_architecture.parameters()]) == len(combined)
        assert sum(p.nelement() for p in self.network_architecture.parameters()) == imported_parameter_count

        state_dict = self.state_dict()
        for i, key in enumerate(state_dict.keys()):
            state_dict[key] = torch.tensor(combined[i])
        self.load_state_dict(state_dict)


# Finite difference parameters
h = 1e-3
# Other parameters
nconfigs=1

# Simulation parameters
nsteps=0
nrep=2
latparam=2.0
nx=nrep
ny=nrep
nz=nrep
ntypes=2
# SNAP options
twojmax=2
m = (twojmax/2)+1
K = int(m*(m+1)*(2*m+1)/6)
print(f"K : {K}")
rcutfac=1.0 #1.0
rfac0=0.99363
rmin0=0
radelem1=1.0
radelem2=1.0
wj1=1.0
wj2=0.96
quadratic=0
bzero=0
switch=0
bikflag=1
dbirjflag=1
#snap_options=f'{rcutfac} {rfac0} {twojmax} {radelem1} {radelem2} {wj1} {wj2} rmin0 {rmin0} quadraticflag {quadratic} bzeroflag {bzero} switchflag {switch}'
snap_options=f'{rcutfac} {rfac0} {twojmax} {radelem1} {radelem2} {wj1} {wj2} rmin0 {rmin0} quadraticflag {quadratic} bzeroflag {bzero} switchflag {switch} bikflag {bikflag} dbirjflag {dbirjflag}'

#print(snap_options)
#lmp = lammps()
lmp = lammps(cmdargs=["-log", "none", "-screen", "none"])

# LAMMPS setup commands

def prepare_lammps():

    lmp.command("clear")
    lmp.command("units metal")
    lmp.command("boundary	p p p")
    lmp.command("atom_modify	map hash")
    lmp.command(f"lattice         bcc {latparam}")
    lmp.command(f"region		box block 0 {nx} 0 {ny} 0 {nz}")
    lmp.command(f"create_box	{ntypes} box")
    lmp.command(f"create_atoms	{ntypes} box")
    lmp.command("mass 		* 180.88")
    lmp.command("displace_atoms 	all random 0.1 0.1 0.1 123456")
    lmp.command(f"pair_style zero 7.0")
    lmp.command(f"pair_coeff 	* *")
    #lmp.command(f"compute 	snap all snap {snap_options}")
    lmp.command(f"compute 	snap all snap {snap_options}")
    lmp.command(f"compute snapneigh all snapneigh {snap_options}")
    lmp.command(f"thermo 		100")



# Get equilibrium position, natoms, number descriptors, length of dbirj
prepare_lammps()
lmp.command(f"run 0")
# These need to be run after run 0 otherwise you'll get a segfault since compute variables don't get initialized.
lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
#print(lmp_snap)
force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
#print(lmp_snap[16:,:])
#print(np.shape(force_indices))
#print(force_indices[0:34,:])
x_indices = force_indices[0::3]
y_indices = force_indices[1::3]
z_indices = force_indices[2::3]
x0 = lmp.numpy.extract_atom("x").flatten()
natoms = lmp.get_natoms()
descriptors = lmp_snap[:natoms, :]
nd = np.shape(descriptors)[1]
dDdR_length = np.shape(lmp_snap)[0]-natoms-6
dDdR = lmp_snap[natoms:(natoms+dDdR_length), :]
#print(np.shape(dDdR)) # Should be same as force_indices
#print(np.shape(force_indices))
# Define indices upon which to contract per-atom energies
indices = []
for m in range(0,nconfigs):
    for i in range(0,natoms):
        indices.append(m)
indices = torch.tensor(indices, dtype=torch.int64)
# Number of atoms per config is needed for future energy calculation.
num_atoms = natoms*torch.ones(nconfigs,dtype=torch.int32)

#Define the network parameters
#layer_sizes = ['num_desc', '10', '8', '6', '1'] # FitSNAP style
print(f"number descriptors: {nd}")
layer_sizes = [nd, nd, nd, 1]

# Build the model
network_architecture = create_torch_network(layer_sizes)
"""
for name, param in network_architecture.named_parameters():
    print("-----")
    print(name)
    print(param)
"""
model = FitTorch(network_architecture, nd)

# Scatter atoms with new position
i = 0
a = 0
atomindx = 3*i+a
n3 = 3*natoms
x = (n3*c_double)()

# Model force

for indx in range(0,n3):
    x[indx]=x0[indx]
#x[atomindx] += h
prepare_lammps()
lmp.scatter_atoms("x",1,3,x)
lmp.command(f"run 0")
#blah = lmp.numpy.extract_atom("x").flatten()
#print(blah)
lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
#print(lmp_snap[16:,:])
#print(force_indices)
# Calculate energy
descriptors = lmp_snap[:natoms, :]
dDdR_length = np.shape(lmp_snap)[0]-natoms-6
dDdR = lmp_snap[natoms:(natoms+dDdR_length), :]
descriptors = torch.from_numpy(descriptors).float().requires_grad_()
dDdR = torch.from_numpy(dDdR).float().requires_grad_()
#print(descriptors)
(energies, forces) = model(descriptors, dDdR, indices, num_atoms, force_indices)
#print(energies)
#e1 = energies.detach().numpy()[0]
forces = forces.detach().numpy()
print(forces[0])


# +h
for indx in range(0,n3):
    x[indx]=x0[indx]
x[atomindx] += h
prepare_lammps()
lmp.scatter_atoms("x",1,3,x)
lmp.command(f"run 0")
#blah = lmp.numpy.extract_atom("x").flatten()
#print(blah)
lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
#print(lmp_snap[16:,:])
#print(force_indices)
# Calculate energy
descriptors = lmp_snap[:natoms, :]
dDdR_length = np.shape(lmp_snap)[0]-natoms-6
dDdR = lmp_snap[natoms:(natoms+dDdR_length), :]
descriptors = torch.from_numpy(descriptors).float().requires_grad_()
dDdR = torch.from_numpy(dDdR).float().requires_grad_()
#print(descriptors)
(energies, forces) = model(descriptors, dDdR, indices, num_atoms, force_indices)
#print(energies)
e1 = energies.detach().numpy()[0]
print(e1)

# -h
for indx in range(0,n3):
    x[indx]=x0[indx]
x[atomindx] -= h
prepare_lammps()
lmp.scatter_atoms("x",1,3,x)
lmp.command(f"run 0")
#blah = lmp.numpy.extract_atom("x").flatten()
#print(blah)
lmp_snap = lmp.numpy.extract_compute("snap",0, 2)
force_indices = lmp.numpy.extract_compute("snapneigh", 0, 2).astype(np.int32)
#print(lmp_snap[16:,:])
#print(force_indices)
# Calculate energy
descriptors = lmp_snap[:natoms, :]
#print(descriptors)
dDdR_length = np.shape(lmp_snap)[0]-natoms-6
dDdR = lmp_snap[natoms:(natoms+dDdR_length), :]
descriptors = torch.from_numpy(descriptors).float().requires_grad_()
dDdR = torch.from_numpy(dDdR).float().requires_grad_()
(energies, forces) = model(descriptors, dDdR, indices, num_atoms, force_indices)
e2 = energies.detach().numpy()[0]
print(e2)

# Finite difference
f_fd = (e1-e2)/(2*h)
print("fd, model:")
print(f_fd)
