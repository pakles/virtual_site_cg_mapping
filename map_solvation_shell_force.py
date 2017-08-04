#!/usr/bin/env python

# Takes a CG mapped trajectory and maps a mirror bead (e.g., to represent the first solvation shell) as the cernter of mass of all solvent beads within a set distance from the base (lipid head group)
# Assign forces based on the inverse mass matrix for overlapping CG mapping
# Inputs: data.lammpstrj atom_index1(base) atom_index2(solvent) num_atoms_per_molecule ShellRad num_CG_atoms_per_frame
# Currently assumes that the system is composed of identical molecules plus solvent and sorted (xyz are in col 3-5)
# Atom indices are based on its position within the molecule topology (first bead is index 0)

import numpy as np
import math as math
import sys as sys

#file details from custom dump trajectory
#log file is now .traj file
if(len(sys.argv) != 7) :
        print "ERROR data.lammpstrj atom_index1 atom_index2 num_atoms_per_molecule ShellRad num_CG_atoms_per_frame"
        exit(0)

name = sys.argv[1] 
aType1 =  int(sys.argv[2])
aType2 = int(sys.argv[3])
nAtomsMol = int(sys.argv[4])
ShellRad = float(sys.argv[5])
nCG = int(sys.argv[6]) # number of actual CG sites per frame (nAtoms - nSolvent)
# Upper and lower bound of the coordinates
Xb = [0, 0]
Yb = [0, 0]
Zb = [0, 0]
# Size of the simulation box
X = 0
Y = 0
Z = 0
# Distance between solvent and base particles
delta = [0, 0, 0]

in_file = open(name,'r')
out_file = open("test.lammpstrj",'w')

all_lines = in_file.readlines()
in_file.close()

nAtoms = int(all_lines[3].split()[0])
nFrames = (len(all_lines)-9)/nAtoms
nMols = nCG/nAtomsMol

print "Analyzing a total of %d frames, each has %d atoms that make up %d molecules\n" % (nFrames,nCG,nMols)
print "Creating %d more atoms per frame to represent the mirror (solvent) bead\n" % nMols
write = out_file.write
# inv_mass = np.zeros(nFrames, nMols, nMols) (might take up too much storage)
# Construct the matrix which stores the inverse mass matrix and the first and second moment of inverse mass matrix of CG sites
inv_mass = np.zeros((nMols, nMols))
first_moment_inv_mass = np.zeros((nMols, nMols)) 
second_moment_inv_mass = np.zeros((nMols, nMols))
# Construct the configuration and momentum mapping matrix for all frames
# config = np.zeros(nFrames, nMols, nAtoms - nCG) (might take up too much storage)
# momentum = np.zeros(nFrames, nMols, nAtoms - nCG) (might take up too much storage)
# Construct the force matrix which stores the 3-d forces on the solvent molecules
force = np.zeros((nAtoms - nCG, 3))
Virtual_vec = np.zeros((nCG, 3))

for i in range(nFrames) :
	# first modify the 9 head lines with new number of atoms and print the rest of the header
	line_index = i*(nAtoms+9)
	header = []
	Xb = np.asarray([float(item) for item in all_lines[line_index + 5].split()])
	Yb = np.asarray([float(item) for item in all_lines[line_index + 6].split()])
	Zb = np.asarray([float(item) for item in all_lines[line_index + 7].split()])
	X = Xb[1] - Xb[0]
	Y = Yb[1] - Yb[0]
	Z = Zb[1] - Zb[0]
	for k in range(9) :   		
		header.append(all_lines[line_index + k])
	newAtoms = nCG + nMols
	# add one virtual site for each molecule as a new 'atom'
	header[3] = str(newAtoms)+"\n"

	for item in header :
		write(item)
		
	# construct a neightbor grid of solvent particles
	sol_index = i*(nAtoms+9) + nMols*nAtomsMol + 9
	# number of cells in each dimension
	Xi = math.floor(X/ShellRad)
	Yj = math.floor(Y/ShellRad)
	Zk = math.floor(Z/ShellRad)
	# size of each cell
	length = X/Xi
	width = Y/Yj
	height = Z/Zk
	# neighgrid[Xi][Yj][Zk][0][0] = num of solvent particles in each cell, neighgrid[Xi][Yj][Zk][n>0] : coordinates of the nth solvent particle in cell (Xi, Yj, Zk)
	neighgrid = np.zeros((Xi+1, Yj+1, Zk+1, nAtoms - nCG, 8))
	
	# construct the configuration CG mapping matrix
	config_map = np.zeros((nMols, nAtoms - nCG))

	for nSol in range(nAtoms - nCG) :
		Sol_bead = np.asarray([float(item) for item in all_lines[sol_index + nSol].split()])
		force[nSol] = np.asarray([float(item) for item in all_lines[sol_index + nSol].split()][5:8])
		Xi_Sol_grid = int(math.floor((Sol_bead[2]-Xb[0])/length))
		Yj_Sol_grid = int(math.floor((Sol_bead[3]-Yb[0])/length))
		Zk_Sol_grid = int(math.floor((Sol_bead[4]-Zb[0])/length))
		neighgrid[Xi_Sol_grid][Yj_Sol_grid][Zk_Sol_grid][0][0] += 1
		counter = neighgrid[Xi_Sol_grid][Yj_Sol_grid][Zk_Sol_grid][0][0]
		neighgrid[Xi_Sol_grid][Yj_Sol_grid][Zk_Sol_grid][counter] = Sol_bead
	# now analyze each molecule, append a molecule type 0 and corresponding forces for the mirror bead
	atom_index = 1
	for j in range(nMols) :
		line_index = i*(nAtoms+9) + j*(nAtomsMol) + 9 # line number for the head group of each molecule
		molecule = []
		SolShell = []
		for k in range(nAtomsMol) :
			line = all_lines[line_index + k]
			molecule.append(line)
		base_vec = np.asarray([float(item) for item in molecule[aType1].split()[2:5]])
		Xi_grid = int(math.floor((base_vec[0]-Xb[0])/length))
		Yj_grid = int(math.floor((base_vec[1]-Yb[0])/length))
		Zk_grid = int(math.floor((base_vec[2]-Zb[0])/length))
		range_x = range_y = range_z = [-1, 0, 1]
		# now iterate through all solvent particles in neightboring cells to find the ones in the first solvation shell
		if  Xi_grid == 0 :
			range_x = [0, 1, Xi]
	       	if  Xi_grid == Xi :
	       		range_x = [-Xi, -1, 0]
	       	if  Yj_grid == 0 :
	       		range_y = [0, 1, Yj]
	       	if  Yj_grid == Yj :
	       		range_y = [-Yj, -1, 0]
	       	if  Zk_grid == 0 :
	       		range_z = [0, 1, Zk]
	       	if  Zk_grid == Zk :
	       		range_z = [-Zk, -1, 0]
		for a in range_x :
	       		for b in range_y :
	       			for c in range_z :
	       				#print(str(Xi_grid)+"\n")
					#print(a)
					for nSol in range(int(neighgrid[Xi_grid + a][Yj_grid + b][Zk_grid + c][0][0])) :
	       					Sol_vec =  neighgrid[Xi_grid + a][Yj_grid + b][Zk_grid + c][nSol+1][2:5]
	       					delta[0] = min(abs(base_vec[0] - Sol_vec[0]), X - abs(base_vec[0] - Sol_vec[0]))
	       					delta[1] = min(abs(base_vec[1] - Sol_vec[1]), Y - abs(base_vec[1] - Sol_vec[1]))
	       					delta[2] = min(abs(base_vec[2] - Sol_vec[2]), Z - abs(base_vec[2] - Sol_vec[2]))
	       					distance = np.linalg.norm(delta)
	       					if distance <= ShellRad :
	       						if X - abs(base_vec[0] - Sol_vec[0]) < abs(base_vec[0] - Sol_vec[0]) :
	       							Sol_vec[0] = Sol_vec[0] + np.sign(base_vec[0] - Sol_vec[0]) * X
	       						if Y - abs(base_vec[1] - Sol_vec[1]) < abs(base_vec[1] - Sol_vec[1]) :
	       							Sol_vec[1] = Sol_vec[1] + np.sign(base_vec[1] - Sol_vec[1]) * Y
	       						if Z - abs(base_vec[2] - Sol_vec[2]) < abs(base_vec[2] - Sol_vec[2]) :
	       							Sol_vec[2] = Sol_vec[2] + np.sign(base_vec[2] - Sol_vec[2]) * Z
							SolShell.append(Sol_vec)
							config_map[j][neighgrid[Xi_grid + a][Yj_grid + b][Zk_grid + c][nSol+1][0] - nCG - 1] = 1
							# print("%f %f %f\n" % (Sol_vec[0], Sol_vec[1], Sol_vec[2]))
		SolShell_mat = np.zeros((len(SolShell),3))
		config_map[j] /= len(SolShell) 
		counter = 0
		for row in SolShell :
			SolShell_mat[counter,:] = row[:]
			counter = counter + 1
		Virtual_vec[j] = np.sum(SolShell_mat, axis = 0)/len(SolShell)
		if Virtual_vec[j][0] > Xb[1] :
			Virtual_vec[j][0] = Virtual_vec[j][0] - X
		if Virtual_vec[j][1] > Yb[1] :
			Virtual_vec[j][1] = Virtual_vec[j][1] - Y
		if Virtual_vec[j][2] > Zb[1] :
			Virtual_vec[j][2] = Virtual_vec[j][2] - Z
		if Virtual_vec[j][0] < Xb[0] :
			Virtual_vec[j][0] = Virtual_vec[j][0] + X
		if Virtual_vec[j][1] < Yb[0] :
			Virtual_vec[j][1] = Virtual_vec[j][1] + Y
		if Virtual_vec[j][2] < Zb[0] :
			Virtual_vec[j][2] = Virtual_vec[j][2] + Z

#		Virtual_vec = np.sum(np.asmatrix([float(num) for num in item.split()[2:5] for item in SolShell]), axis = 0)/len(SolShell)
#		print Virtual_vec
#	base_vec = np.asarray([float(item) for item in molecule[aType1].split()[2:5]])
#       dir_vec = np.asarray([float(item) for item in molecule[aType2].split()[2:5]])
#       mirror_vec = (base_vec - dir_vec)
#       mirror_norm = np.linalg.norm(mirror_vec)
#       mirror_vec = mirror_vec * mirrorLen / mirror_norm

	#       write new lines
		
	inv_mass = np.dot(config_map, np.transpose(config_map))
	first_moment_inv_mass += inv_mass

	for row in range(nMols) :
		for column in range(nMols) :
			second_moment_inv_mass[row][column] += np.square(inv_mass[row][column])
        
	momentum_map = np.linalg.inv(inv_mass).dot(config_map)	       
	force_CG = np.dot(momentum_map, force)
        print(force_CG)

	for j in range(nMols) :
		out_file.write("%d 0 %f %f %f %f %f %f\n" % (atom_index, Virtual_vec[j][0], Virtual_vec[j][1], Virtual_vec[j][2], force_CG[j][0], force_CG[j][1], force_CG[j][2]))
		atom_index = atom_index + 1
		molecule = []
		line_index = i*(nAtoms+9) + j*(nAtomsMol) + 9
		for k in range(nAtomsMol) :
			line = all_lines[line_index + k]
			molecule.append(line)
		for row in molecule :
			data = row.split()
			data[0] = str(atom_index)
			atom_index = atom_index + 1
			out_file.write(" ".join(data)+"\n")

first_moment_inv_mass /= nFrames
second_moment_inv_mass /= nFrames

out_file.close()

out_file = open("inv_mass.lammpstrj", 'w')
out2_file = open("inv_mass2.lammpstrj", 'w')

out_file.write('#The first moment of the inverse mass matrix is:\n')

for row in range(nMols) :
	for column in range(nMols) :
        #out_file.write(str(first_moment_inv_mass[row][column])+ ' ')
		out_file.write("%d %d %f\n" % (row, column, first_moment_inv_mass[row][column]))

out2_file.write('The second moment of the inverse mass matrix is:\n')

for row in range(nMols) :
	for column in range(nMols) :
		out2_file.write("%d %d %f\n" % (row, column, second_moment_inv_mass[row][column]))

