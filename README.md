
# Kinetic Monte Carlo -simulation of surface growth
--------------------------------------------------------------------------------
This is a python script to simulate surface growth of an arbitrary material
with periodic step boundary conditions.

The goal of this simulation was to see how temperature affects the coarseness
when growing a surface, and to find how it would be possible to grow a smoother
surface.

The surface simulated is a {100} lattice where in the output, lattice repeats
periodically in the left–right direction and periodically from top to down as
steps as follows

	--------
	        --------
	                --------

Output can be shown in real time during simulation, or be saved as mp4 by
changing the save variable from beginning of .py file.


## Parameters
--------------------------------------------------------------------------------

Parameters relating to the lattice such as interaction energies, and relating
to the conditions such as tempersture can be changes in the beginning of the
python file.


## Outputs
--------------------------------------------------------------------------------

The code outputs .png and .mp4 files with variable save set as True. Both file
types show the surface with color indicating the height of the surface. From
output two measures, coverage and variance can also be seen.

Coverage is presented as a presentage and indicates how many adatoms are
adsorbed by the surface, with 100% indicating that enough adatoms have been
adsorbed to fill exactly one layer of the surface.

Variance on the other hand indicates the variance in height of the surface.
This is used as a measure of coarseness of the surface.

