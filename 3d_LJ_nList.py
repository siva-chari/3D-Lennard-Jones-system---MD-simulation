# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 13:41:55 2021

@author: Archana P S
"""
# 3d lattice

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numba import jit

plt.rcParams['font.family']="Times New Roman"
plt.rcParams['xtick.labelsize']=18
plt.rcParams['ytick.labelsize']=18


#=================== define all the parameters here. =================#

nx = 5				# no. of atoms along each direction of the cube.
N = nx**3			# total no. of atoms.
rho = 0.5			# desired density.
temperature = 1.0	# desired temperature.

sigma = 1.0
epsilon = 1.0
mass = 1.0

rcut = 2.5*sigma
vskin = 0.5*sigma

dt = 0.002
nmaxsteps = 5000
thermo_freq = 10
nbrList_freq = 5


#=====================================================================#


@jit
def put_on_3d_lattice(N, rho, sigma):
	lx = (N/rho)**(1/3)
	ly = lx
	lz = lx
	nx = int(np.cbrt(N))
	ny = nx
	nz = nx
	dx = (lx - nx*sigma)/(nx-1)
	x = np.zeros(N)
	y = np.zeros(N)
	z = np.zeros(N)
	ix = 0
	iy = 0
	iz = 0
	for i in range(N):
		if (i % nx == 0):
			ix = 0
			if (i % (nx)**2 == 0):
				iy = 0
				if (i % (nx)**3 == 0):
					iz = 0
				else:
					iz = iz +1
			else:
				iy = iy + 1
		else:
			ix = ix + 1
		
		x[i] = sigma/2.0 + ix*(dx + sigma)			
		y[i] = sigma/2.0 + iy*(dx + sigma)
		z[i] = sigma/2.0 + iz*(dx + sigma)
		
	return [x,y,z,lx,ly,lz]



def write_xyz_file(filename,x,y,z):
	fout_xyz = open(filename, 'w+')
	
	nMax = x.size
	
	fout_xyz.write("{}\n".format(nMax))
	fout_xyz.write("comment\n")
	for i in range(nMax):
		fout_xyz.write("1 {} {} {}\n".format(x[i], y[i], z[i]))
	fout_xyz.close()
	return


@jit
def computeForces(x,y,z,natoms,sigma,epsilon):
	fx[:] = 0.0
	fy[:] = 0.0
	fz[:] = 0.0
	
	PE = 0.0
	virial = 0.0
	
	for i in range(natoms):
		for j in range(natoms):
			#avoid the self interaction.
			if (j != i):
				#calculate distance b/w i and j particles.
				dx = x[i] - x[j]
				dy = y[i] - y[j]
				dz = z[i] - z[j]
				
				# minimum image convention.
				dx = dx - np.round(dx/lx)*lx
				dy = dy - np.round(dy/ly)*ly
				dz = dz - np.round(dz/lz)*lz
				
				# distance b/w i and j particles.
				dr = np.sqrt(dx**2 + dy**2 + dz**2)
				
				# now calculate the force.
				sr6 = (sigma/dr)**6.0
				rinv = 1.0/dr
				rinv2 = rinv**2.0
				comn_frc_term = 48.0*epsilon*sr6*(sr6 - 0.5)*rinv2
				fx[i] = fx[i] + comn_frc_term*dx
				fy[i] = fy[i] + comn_frc_term*dy
				fz[i] = fz[i] + comn_frc_term*dz
				
				# calculate potential energy here.
				pot_term = 4.0*epsilon*sr6*(sr6 - 1.0)
				PE = PE + pot_term
				
				# calculation of virial.
				vir_term = dx*fx[i] + dy*fy[i] + dz*fz[i]
				virial = virial + vir_term
	
	PE = PE * 0.5
	virial = virial * 0.5
	
	return [fx,fy,fz,PE,virial]


@jit
def VelocityVerlet_step_1(x,y,z,vx,vy,vz,fx,fy,fz,N,dt,mass):
    # this does the first step of V-V algorithm.
    for i in range(N):
        # position update
        x[i] = x[i] + vx[i]*dt + 0.5*fx[i]/mass * dt**2.0
        y[i] = y[i] + vy[i]*dt + 0.5*fy[i]/mass * dt**2.0
        z[i] = z[i] + vz[i]*dt + 0.5*fz[i]/mass * dt**2.0
        # velocity update.
        vx[i] = vx[i] + fx[i]*dt*0.5
        vy[i] = vy[i] + fy[i]*dt*0.5
        vz[i] = vz[i] + fz[i]*dt*0.5
    return [x,y,z,vx,vy,vz]


@jit
def VelocityVerlet_step_2(vx,vy,vz,fx,fy,fz,N,dt,mass):
    # update only velocities. and calculate Kinetic energy.
    KE = 0.0
    for i in range(N):
        vx[i] = vx[i] + fx[i]*dt*0.5
        vy[i] = vy[i] + fy[i]*dt*0.5
        vz[i] = vz[i] + fz[i]*dt*0.5
        
        KE = KE + (vx[i]**2.0 + vy[i]**2.0 + vz[i]**2.0)*mass*0.5
    return [vx,vy,vz,KE]


#======== function which will calculate the neighbor list.
@jit
def get_Neighbor_List(natoms,x,y,z,lx,ly,lz,sigma,rcut,vskin):	# Siva, 19 Sept, 2021.

	Distances = np.zeros((natoms,natoms))

	nCount[:] = 0
	nList[:,:] = 0

	for i in range(natoms):
		Distances[i,i] = lx
		
		for j in range(natoms):
			if(j != i):
				dx = x[i] - x[j]
				dy = y[i] - y[j]
				dz = z[i] - z[j]
				#minimum image convention.
				dx = dx - np.round(dx/lx)*lx
				dy = dy - np.round(dy/ly)*ly
				dz = dz - np.round(dz/lz)*lz
				
				rij = np.sqrt(dx**2 + dy**2 + dz**2)
				Distances[i,j] = rij
				#Distances[j,i] = Distances[i,j]
				
				verlet_R = (rcut+vskin)*sigma
				
				if(rij < verlet_R):
					nCount[i] = nCount[i]+1
					k = nCount[i]
					# start_index = i*natoms
					nList[i, k-1] = j
				else:
					continue

	return [nCount,nList,Distances]

#======== function which will compute the forces on all the particles,
#======== using the list of neighbors for every particle.

@jit
def compute_Forces_nbrList(natoms,x,y,z,nCount,nList,sigma,epsilon,lx,ly,lz,fx,fy,fz):	# Siva, 19 Sept, 2021.
	fx[:] = 0.0
	fy[:] = 0.0
	fz[:] = 0.0

	PE = 0.0
	virial = 0.0

	for i in range(natoms):
		#
		for k in range(nCount[i]):
			#starting = i*natoms
			j = nList[i, k]
			#
			if(j != i):
				#calculate the distance
				dx = x[i]-x[j]
				dy = y[i]-y[j]
				dz = z[i]-z[j]
				#minimum  image.
				dx = dx - np.round(dx/lx)*lx
				dy = dy - np.round(dy/ly)*ly
				dz = dz - np.round(dz/lz)*lz
				
				rij = np.sqrt(dx**2.0 + dy**2.0 + dz**2.0)
				rij2 = rij**2.0
				rcut2 = rcut**2.0
				
				if(rij2 < rcut2):
					# need to calculate the force.
					rinv = 1.0/rij
					rinv2 = rinv**2.0
					sr6 = (sigma/rij)**6.0
					src6 = (sigma/rcut)**6.0
					rcinv = 1.0/rcut
					rcinv2 = rcinv**2.0
					#
					#use LJ potential, with predefined cut-off.
					frc_common = 48.0*epsilon*sr6*(sr6 - 0.5)*rinv2
					fx[i] = fx[i] + frc_common*dx
					fy[i] = fy[i] + frc_common*dy
					fz[i] = fz[i] + frc_common*dz
					# shifting for the potential force-shifting.
					frc_shift = 48.0*epsilon*src6*(src6 - 0.5)*rcinv2
					fx_shift = frc_shift*dx
					fy_shift = frc_shift*dy
					fz_shift = frc_shift*dz
					#shift it.
					fx[i] = fx[i] - fx_shift
					fy[i] = fy[i] - fy_shift
					fz[i] = fz[i] - fz_shift
					# now calculate PE & virial.
					pot_lj = 4.0*epsilon*sr6*(sr6 - 1.0)
					pot_rc = 4.0*epsilon*src6*(src6 - 1.0)
					pot_fs = -48.0*epsilon*src6*(src6 - 0.5)*rcinv
					# add all the components./ shifting.
					PE = PE + pot_lj - pot_rc - (rij - rcut)*pot_fs
					virial = virial + (dx*fx[i] + dy*fy[i] + dz*fz[i])
				else:
					continue
	PE = PE*0.5
	virial = virial*0.5
	#
	return [fx,fy,fz,PE,virial]	


@jit
def applyPBC(N,x,y,z,lx,ly,lz):
	x = x - np.round(x/lx)*lx
	y = y - np.round(y/ly)*ly
	z = z - np.round(z/lz)*lz
	
	return [x,y,z]


#======== main program ================================================#
#======== main program ================================================#
#======== main program ================================================#

x = np.zeros(N)	
y = np.zeros(N)
z = np.zeros(N)

vx = np.zeros(N)
vy = np.zeros(N)
vz = np.zeros(N)

vx = np.random.rand(N)
vy = np.random.rand(N)
vz = np.random.rand(N)

fx = np.zeros(N)
fy = np.zeros(N)
fz = np.zeros(N)

nCount = np.zeros(N, dtype=int)
nList = np.zeros((N,N), dtype=int)


[x,y,z,lx,ly,lz] = put_on_3d_lattice(N, rho, sigma)

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z,s=60,color='blue')
# plt.show()

# to write the xyz file.
xyz_file = "out_config.xyz"

write_xyz_file(xyz_file, x,y,z)


#open the thermo file.
thermo_file = "out_thermo.dat"
fout_thermo = open(thermo_file, 'w+')

# get the neighbor list.
[nCount,nList,Distances] = get_Neighbor_List(N,x,y,z,lx,ly,lz,sigma,rcut,vskin)

# now compute the forces, using the neighbor list.
[fx,fy,fz,PE,virial] = compute_Forces_nbrList(N,x,y,z,nCount,nList,sigma,epsilon,lx,ly,lz,fx,fy,fz)

# move the particles by integrating the eq. of motion/ using V.V.
# 1st step of V-V.
[x,y,z,vx,vy,vz] = VelocityVerlet_step_1(x,y,z,vx,vy,vz,fx,fy,fz,N,dt,mass)

# compute forces for the 2nd step of V-V.
[fx,fy,fz,PE,virial] = compute_Forces_nbrList(N,x,y,z,nCount,nList,sigma,epsilon,lx,ly,lz,fx,fy,fz)

#2nd step, of V-V.
[vx,vy,vz,KE] = VelocityVerlet_step_2(vx,vy,vz,fx,fy,fz,N,dt,mass)

fig1 = plt.figure()
fig1.patch.set_facecolor('white')
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x,y,z,s=60,color='blue')
# plt.show()

for itr in range(nmaxsteps):
	timeNow = itr*dt
	
	[x,y,z] = applyPBC(N,x,y,z,lx,ly,lz)
	
	# calculate the neighbor list at a defined interval, after every nbrList_freq steps.
	if (itr % nbrList_freq == 0):
		[nCount,nList,Distances] = get_Neighbor_List(N,x,y,z,lx,ly,lz,sigma,rcut,vskin)

	[x,y,z,vx,vy,vz] = VelocityVerlet_step_1(x,y,z,vx,vy,vz,fx,fy,fz,N,dt,mass)

	#[fx,fy,fz,PE,virial] = computeForces(x,y,z,N,sigma,epsilon)

	[fx,fy,fz,PE,virial] = compute_Forces_nbrList(N,x,y,z,nCount,nList,sigma,epsilon,lx,ly,lz,fx,fy,fz)

	[vx,vy,vz,KE] = VelocityVerlet_step_2(vx,vy,vz,fx,fy,fz,N,dt,mass)
	

	tempInst = KE*2.0/(3.0*N-1)
	virial = virial/(3.0*N)
	pressure = rho*(tempInst + virial)


	if (itr % thermo_freq == 0):
		fout_thermo.write("{} {} {} {} {} {}\n".format(timeNow,tempInst,PE,KE,pressure,virial))
		fout_thermo.flush()


fout_thermo.close()


#=================== to plot.
thermo_data = np.loadtxt('out_thermo.dat')
time = thermo_data[:,0]
T = thermo_data[:,1]
PE = thermo_data[:,2]
KE = thermo_data[:,3]
pressure = thermo_data[:,4]
virial = thermo_data[:,5]

plt.figure(figsize=[5,5])
plt.plot(time,PE,label='PE')
plt.plot(time,KE,label='KE')
plt.plot(time,PE+KE,label='Total E')
plt.title("PE, KE and Total energy",fontsize=18)
plt.xlabel('time t (r.u.)',fontsize=18)
plt.ylabel('Energy (r.u.)',fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig('./forceShifted_E_vs_time.png',format='png',dpi=300)

plt.figure(figsize=[5,5])
plt.plot(time,T)
plt.title("Temperature",fontsize=18)
plt.xlabel('time t (r.u.)',fontsize=18)
plt.ylabel('Temperature (r.u.)',fontsize=18)
plt.tight_layout()
#plt.show()
plt.savefig('./forceShifted_T_vs_time.png',format='png',dpi=300)

plt.figure(figsize=[5,5])
plt.plot(time,pressure,label='Pressure')
plt.plot(time,virial,label='Virial')
plt.title("Pressure and Virial",fontsize=18)
plt.xlabel('time t (r.u.)',fontsize=18)
plt.ylabel('Pressure (r.u.)',fontsize=18)
plt.legend(loc='best',fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig('./forceShifted_P_vs_time.png',format='png',dpi=300)
