#!/usr/bin/env python

# An attempt to generalize the mapping script for multi-component systems
# Also recognizes that if each CG site has at least one uniquely mapped molecule, we might
# be able to use Eq. 27 in Noid et al., JCP 2008, 244144.
# Will also sort the trajectory if needed

# Inputs:
#    data.lammpstrj - LAMMPS trajectory file. The molecules do not need to be sorted but the order within each molecule needs to be consiste                      nt. The head group must always come first
#    list_head_groups - A list of atom types used to define each head group and water (water is always last), e.g., "1 5 8 18"
#    list_number_of_molecules - A list of the number of copies of each molecule, e.g., "192 128 34 1568"
#    list_atoms_per_molecule - A list of the number of CG sites per molecule, e.g., "6 8 3 1"
#    list_find_vcg - A list of boolean flags (0's and 1's) to decide if a molecule needs a VCG site, e.g., "1 1 0 0"
#    solvation_radius - The radius within which each VCG site is computed
#    neighbor_bin_size - The length of the cell list used to accelerate neighbor searches (> solvation_radius)

# Outputs:
#    vcg_mapped.lammpstrj - A LAMMPS trajectory (sorted such that all similar molecules are grouped together) with VCG sites. The VCG sites                             have atom types that increment starting from the largest atom type

import numpy as np
import math as math
import sys as sys

# some auxilary functions
def get_wrapped_coords(posarr, Xb, Yb, Zb) :
        xx = posarr[0]
        yy = posarr[1]
        zz = posarr[2]

        lenX = Xb[1] - Xb[0]
        lenY = Yb[1] - Yb[0]
        lenZ = Zb[1] - Zb[0]

        while(xx < Xb[0]) :
                xx += lenX
        while(xx > Xb[1]) :
                xx -= lenX
        while(yy < Yb[0]) :
                yy += lenY
        while(yy > Yb[1]) :
                yy -= lenY
        while(zz < Zb[0]) :
                zz += lenZ
        while(zz > Zb[1]) :
                zz -= lenZ
                
        return np.asarray([xx, yy, zz])

def get_minimum_image(refarr, posarr, lenX, lenY, lenZ) :
        rx = refarr[0]
        ry = refarr[1]
        rz = refarr[2]
        xx = posarr[0]
        yy = posarr[1]
        zz = posarr[2]

        while(abs(rx - xx) > lenX/2.0) :
                xx = xx + np.sign(rx-xx)*lenX
        while(abs(ry - yy) > lenY/2.0) :
                yy = yy + np.sign(ry-yy)*lenY
        while(abs(rz - zz) > lenZ/2.0) :
                zz = zz + np.sign(rz-zz)*lenZ

        return np.asarray([xx,yy,zz])

def get_nnlist_kernel(refarr, posarr, lenX, lenY, lenZ) :
        return true
        
def get_neigh_grid_bins(posarr, Xb, Yb, Zb, nX, nY, nZ, neigh_bin) :
        x_grid = int(math.floor((posarr[0]-Xb[0])/neigh_bin))
        y_grid = int(math.floor((posarr[1]-Yb[0])/neigh_bin))
        z_grid = int(math.floor((posarr[2]-Zb[0])/neigh_bin))
        # fix potential floating point errors
        if x_grid < 0 :
                x_grid = 0
        if x_grid >= nX :
                x_grid = nX-1
        if y_grid < 0 :
                y_grid = 0
        if y_grid >= nY :
                y_grid = nY-1
        if z_grid < 0 :
                z_grid = 0
        if z_grid >= nZ :
                z_grid = nZ-1       
        return x_grid, y_grid, z_grid

#file details from custom dump trajectory
#log file is now .traj file
if(len(sys.argv) != 8) :
        print "ERROR data.lammpstrj list_head_groups list_number_of_molecules list_atoms_per_molecule list_find_vcg solvation_radius neighbor_bin_size"
        exit(0)

# ----- GATHER ALL USER INPUTS HERE ----------------------------------------
trajname = sys.argv[1] 
outname = "vcg_mapped.lammpstrj"

head_list = [int(i) for i in sys.argv[2].split()]
nmol_list =  [int(i) for i in sys.argv[3].split()]
natom_list = [int(i) for i in sys.argv[4].split()]
vcg_flag_list = [int(i) for i in sys.argv[5].split()]

solv_rad = float(sys.argv[6])
neigh_bin = float(sys.argv[7])
# --------------------------------------------------------------------------

# ----- INITIALIZE SOME VARIABLES HERE -------------------------------------
Xb = [0, 0] #xlo, xhi
Yb = [0, 0] #ylo, yhi
Zb = [0, 0] #zlo, zhi
lenX = 0
lenY = 0
lenZ = 0
dr = np.zeros(3) # placeholder for r_ij vector
nAtoms = 0
nMolsFinal = 0
nAtomsFinal = 0
nAtomsFinal_noVCG = 0
nVCG = 0
nSol = 0
largest_atype = 0
# --------------------------------------------------------------------------

traj_file = open(trajname,'r')
all_lines = traj_file.readlines()
traj_file.close()
#out_file = open("ExplicitForce_n=%d.lammpstrj" % nFramesAnalyzed,'w')
out_file = open(outname, 'w')

nAtoms = int(all_lines[3].split()[0])
nFrames = len(all_lines) / ( nAtoms + 9 )
# ----- SOME BASIC SANITY CHECKS HERE --------------------------------------

# the number of list elements should be the same
if(len(head_list) != len(nmol_list) or len(head_list) != len(natom_list) or len(head_list) != len(vcg_flag_list)) :
        print "Inconsistency detected in list definitions. Please check your inputs. Exiting..."
        exit(1)

# the number of non-solvent + solvent CG atoms should be less than the number of atoms in your trajectory
nSol = nmol_list[-1] * natom_list[-1]
for i in range(len(head_list)-1) :
        nMolsFinal = nMolsFinal + nmol_list[i]
        nAtomsFinal_noVCG = nAtomsFinal_noVCG + nmol_list[i] * natom_list[i]
        if (vcg_flag_list[i] == 1) :
                nAtomsFinal = nAtomsFinal + nmol_list[i] * ( 1 + natom_list[i] )
        else :
                nAtomsFinal = nAtomsFinal + nmol_list[i] * ( natom_list[i] )
nVCG = nAtomsFinal - nAtomsFinal_noVCG

if( (nAtomsFinal_noVCG + nSol) > nAtoms ) :
        print "Your molecule statistics are inconsistent with the available atoms in your trajectory file. Please check your inputs. Exiting..."
        exit(1)
# --------------------------------------------------------------------------

print "Analyzing a total of %d frames, each has %d atoms but will be reduced to %d atoms (without virtual sites)\n" % (nFrames,nAtoms,nAtomsFinal_noVCG)
print "Creating %d total atoms per frame (for %d molecules total) to represent solvent beads\n" % (nAtomsFinal, nMolsFinal)
print "Using %d VCG sites\n" % (nVCG)
write = out_file.write

for i in range(nFrames) :
        # for every frame, reset data
        # create a LIST of matrices containing type/position/force information for each molecule type; atomtype, molID, x, y, z, fx, fy, fz
        cg_sites = [] #for cg-mapped molecules (non-solvent)
        virtual_sites = [] #for new vcg-mapped solvent
        for j in range(len(head_list)-1) :
                #cg_sites.append(np.zeros(nmol_list[i]*natom_list[i],8)))
                #virtual_sites.append(np.zeros(nmol_list[i]*vcg_flag_list[i],8)))
                cg_sites.append([])
                virtual_sites.append([])
        
        solvent_sites = np.zeros((nSol,8))
        solvent_assignment = np.zeros((nSol,1000)) #first index is the number of assigned VCG sites, the rest is the list of VCG indices; this will be used to determine dIJ and cIJ as well
        VCG_assignment = np.zeros((int(len(head_list)-1),nAtomsFinal_noVCG,1000)) #first index is the number of assigned solvent sites, the rest is the list of SOLVENT indices

	# first modify the 9 head lines with new number of atoms and print the rest of the header
	line_index = (i)*(nAtoms+9)
        print "Starting from line %d" % line_index
	header = []
	Xb = np.asarray([float(item) for item in all_lines[line_index + 5].split()], dtype=np.float64)
	Yb = np.asarray([float(item) for item in all_lines[line_index + 6].split()], dtype=np.float64)
	Zb = np.asarray([float(item) for item in all_lines[line_index + 7].split()], dtype=np.float64)
	lenX = Xb[1] - Xb[0]
	lenY = Yb[1] - Yb[0]
	lenZ = Zb[1] - Zb[0]
	for k in range(9) :   		
                header.append(all_lines[line_index + k])
	header[3] = str(nAtomsFinal)+"\n"

	for item in header :
		write(item)
    
	# construct a neighbor grid of solvent particles
	sol_index = line_index + nAtomsFinal_noVCG + 9
	# max index of cells in each dimension
	nX = int(math.ceil(lenX/neigh_bin))
	nY = int(math.ceil(lenY/neigh_bin))
	nZ = int(math.ceil(lenZ/neigh_bin))
        print "Computing frame %d" % i
        print "Box dimensions: L:%f  W:%f  H:%f" % (lenX, lenY, lenZ)
        print "Neighbor search grid dimensions: L:%d  W:%d  H:%d" % (nX, nY, nZ)
	# neighgrid[nX][nY][nZ][0] = num of solvent particles in each cell, neighgrid[Xi][Yj][Zk][n>0] : index of solvent particles within this cell
	neighgrid = np.zeros((nX, nY, nZ, nmol_list[-1]+1), dtype=np.int) #x, y, z, list size[0] and list of solvent indicies (solvent data in solvent_sites)
        print neighgrid.shape
        # process all solvent molecules first
        # populate its index in neighgrid and put its data into solvent_sites
	for iSol in range(nSol) :
		solvent_sites[iSol,:] = np.asarray([float(item) for item in all_lines[sol_index + iSol].split()], dtype=np.float64)
                # if unwrapped, solvent site coordinates should be rewrapped
                solvent_sites[iSol,2:5] = get_wrapped_coords(solvent_sites[iSol,2:5], Xb, Yb, Zb)
                x_grid, y_grid, z_grid = get_neigh_grid_bins(solvent_sites[iSol,2:5], Xb, Yb, Zb, nX, nY, nZ, neigh_bin)
                neighgrid[x_grid][y_grid][z_grid][0] += 1
		counter = neighgrid[x_grid][x_grid][z_grid][0]
		neighgrid[x_grid][y_grid][z_grid][counter] = iSol

        # process all molecules and assign to cg_sites [list of lists, first level contains each type of molecule]
        # molecules can be out of order so we need to process line-by-line (id, type, x, y, z, fx, fy, fz)
        mol_type_index = -1
        j = 0
        while (j < nAtomsFinal_noVCG) :
                atom_index = line_index + 9 + j
                temp_cg_site = np.asarray([float(item) for item in all_lines[atom_index].split()], dtype=np.float64)
                atype = int(temp_cg_site[1])
                # check atom type for viable head group
                if (atype in head_list) :
                        mol_type_index = head_list.index(atype)
                natom_this_mol = natom_list[mol_type_index]
                for k in range(natom_this_mol) :
                        atom_index = line_index + 9 + j
                        temp_cg_site = np.asarray([float(item) for item in all_lines[atom_index].split()], dtype=np.float64)
                        atype = int(temp_cg_site[1])
                        if (atype > largest_atype) :
                                largest_atype = atype
                        cg_sites[mol_type_index].append(temp_cg_site)
                        j+=1

        # ANOTHER SANITY CHECK TO MAKE SURE THESE WERE PROCESSES CORRECTLY
        for j in range(len(head_list)-1) :
                nmol_this_mol = len(cg_sites[j])
                nmol_compare = nmol_list[j]*natom_list[j]
                if(nmol_compare != nmol_this_mol) :
                        print "ERROR! There is a mismatch in the processed/expected molecule of type %d. Exiting now..." % j
                        print "We found %d atoms but expected %d atoms" % (nmol_this_mol, nmol_compare)
                        exit(1)
                        
	# now analyze each molecule and assign solvent IDs for each relevant molecule's headgroup
        # assignments are stored in solvent_assignment[id]; here, [0] = # of assignments, [n>0] = list of assigned molecule (starting from zero = j)

	for mid, mollist in enumerate(cg_sites) :
                for aid, atominfo in enumerate(mollist) :
                        #if this is not a head-type, we skip
                        atype = int(atominfo[1])
                        if (atype not in head_list) :
                                continue
                        hx = atominfo[2]
                        hy = atominfo[3]
                        hz = atominfo[4]
                        hx_grid, hy_grid, hz_grid = get_neigh_grid_bins(atominfo[2:5], Xb, Yb, Zb, nX, nY, nZ, neigh_bin)

                        range_x = range_y = range_z = [-1, 0, 1]
                        # now iterate through all solvent particles in neighboring cells to find the ones in the solvation shell
                        if  hx_grid == 0 :
                                range_x = [0, 1, nX-1]
                        if  hx_grid == (nX-1) :
                                range_x = [-nX+1, -1, 0]
                        if  hy_grid == 0 :
                                range_y = [0, 1, nY-1]
                        if  hy_grid == (nY-1) :
                                range_y = [-nY+1, -1, 0]
                        if  hz_grid == 0 :
                                range_z = [0, 1, nZ-1]
                        if  hz_grid == (nZ-1) :
                                range_z = [-nZ+1, -1, 0]

                        for a in range_x :
                                for b in range_y :
                                        for c in range_z :
                                                for iSol in range(int(neighgrid[hx_grid + a][hy_grid + b][hz_grid + c][0])) :
                                                        solv_id = neighgrid[hx_grid + a][hy_grid + b][hz_grid + c][iSol+1]
                                                        sx = solvent_sites[solv_id][2]
                                                        sy = solvent_sites[solv_id][3]
                                                        sz = solvent_sites[solv_id][4]

                                                        dr[0] = min(abs(hx-sx), lenX - abs(hx-sx))
                                                        dr[1] = min(abs(hy-sy), lenY - abs(hy-sy))
                                                        dr[2] = min(abs(hz-sz), lenZ - abs(hz-sz))
                                                        distance = np.linalg.norm(dr)
                                                        if distance <= solv_rad :
                                                                # print "We have found a solvent bead for mid %d and aid %d" % (mid, aid)
                                                                # assign this solvent to mid
                                                                solvent_assignment[solv_id][0] += 1
                                                                solv_assign_list_id = int(solvent_assignment[solv_id][0])
                                                                solvent_assignment[solv_id][solv_assign_list_id] = aid
                                                                VCG_assignment[mid][aid][0] += 1
                                                                vcg_assign_list_id = int(VCG_assignment[mid][aid][0])
                                                                VCG_assignment[mid][aid][vcg_assign_list_id] = solv_id

        # after solvent assignment, check for uniqueness
        nUniqueSolv = 0
        nNonAssigned = 0
        for iSol, solvinfo in enumerate(solvent_assignment) :
                if (solvinfo[0] == 1) :
                        nUniqueSolv += 1
                if (solvinfo[0] == 0) :
                        nNonAssigned += 1
        if(nUniqueSolv >= nVCG) :
                print "We have uniqueness. There are %d VCG sites and %d uniquely mapped solvent sites" % (nVCG, nUniqueSolv)
                print "We also have %d solvent that were not assigned." % (nNonAssigned)

        # Now we can construct and store virtual sites (in virtual_sites list)
	for mid, mollist in enumerate(cg_sites) :
                for aid, atominfo in enumerate(mollist) :
                        #if this is not a head-type, we skip
                        atype = int(atominfo[1])
                        if (atype not in head_list) :
                                continue        
                        nAssigned = int(VCG_assignment[mid][aid][0])
                        assigned_solv_list = VCG_assignment[mid][aid][1:(nAssigned+1)]
                        #for now we will just naively sum and average the forces
                        vcg_pos_vec = np.zeros(3)
                        vcg_force_vec = np.zeros(3)
                        head_pos_vec = atominfo[2:5]

                        for sid in assigned_solv_list :

                                # Adjust solvent position to be closest to head group due to PBC
                                sid = int(sid)
                                temp_pos_vec = solvent_sites[sid][2:5]
                                temp_pos_vec = get_minimum_image(head_pos_vec, temp_pos_vec, lenX, lenY, lenZ)

                                vcg_pos_vec = np.add(vcg_pos_vec[:],temp_pos_vec[:])
                                vcg_force_vec = np.add(vcg_force_vec[:],solvent_sites[sid][5:8])

                        if(nAssigned == 0) :
                                print "ERROR: Why do we have a VCG site with zero assignment? Something has gone wrong..."
                                print "This is for molecule type %d and atom id %d with atom type %d" % (mid, aid, atype)
                                exit(1)
                        vcg_pos_vec[:] /= float(nAssigned)
                        virtual_sites[mid].append([vcg_pos_vec, vcg_force_vec])

        # now write all of the new sites to the new output file
        atom_counter = 1
        vcg_type = largest_atype
	for mid, mollist in enumerate(cg_sites) :
                vcg_counter = 0
                if(len(virtual_sites[mid]) > 0) :
                        vcg_type += 1
                for aid, atominfo in enumerate(mollist) :
                        atype = int(atominfo[1])
                        #if this is a VCG candidate and this is the head type, we should output the VCG particle first; then we print the normal CG sites
                        if (atype in head_list) :
                                vcg_pos = virtual_sites[mid][vcg_counter][0]
                                vcg_force = virtual_sites[mid][vcg_counter][1]
                                out_file.write("%d %d %f %f %f %f %f %f\n" % (atom_counter, vcg_type, vcg_pos[0], vcg_pos[1], vcg_pos[2], vcg_force[0], vcg_force[1], vcg_force[2]))
                                vcg_counter+=1
                                atom_counter+=1
                        cgx = atominfo[2]
                        cgy = atominfo[3]
                        cgz = atominfo[4]
                        cgfx = atominfo[5]
                        cgfy = atominfo[6]
                        cgfz = atominfo[7]

                        out_file.write("%d %d %f %f %f %f %f %f\n" % (atom_counter, atype, cgx, cgy, cgz, cgfx, cgfy, cgfz))
                        atom_counter+=1

out_file.close
