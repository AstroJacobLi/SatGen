################################ SubEvo #################################

# Program that evolves the subhaloes intialized by TreeGen_Sub.py
# This version of the code is meant to work with the Green model of
# stripped subhalo density profiles.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech
# Sheridan Beckwith Green 2020 Yale University
# -- Changed loop order so that redshift is the outermost loop,
#    which enables mass of ejected subhaloes to be removed from
#    the corresponding host; necessary for mass conservation

# Jiaxuan Li: this is the script used for ELVES-Dwarf project, paper 2 (Rvir is Rvir, not R200c). Paper1 used "SatEvo.py"
# Difference: now it uses a mass ratio as mass resolution limit, instead of a fixed 1e7 Mres in paper1. 
# Also the stripping efficiency is a function of concentration ratio instead of a fixed value (in paper 1 we had StrippingEfficiency = 0.6)
# The output also contains the tidal radius and the alpha values at each step, which are not in paper1.

######################## set up the environment #########################

import config as cfg
import cosmo as co
import evolve as ev
from profiles import NFW,Green
from orbit import orbit
import aux

import numpy as np
import sys
import os 
import time 
import re
from multiprocessing import Pool, cpu_count

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
#warnings.simplefilter('always', UserWarning)
warnings.simplefilter("ignore", UserWarning)

########################### user control ################################


datadir = "/scratch/gpfs/JENNYG/jiaxuanl/SatGen/OUTPUT_TREE/"
# datadir = '/scratch/gpfs/MERIAN/user/jiaxuanl/SatGen/OUTPUT_TREE/'
outdir = "/scratch/gpfs/JENNYG/jiaxuanl/SatGen/OUTPUT_SAT/"

Rres_factor = 10**-3 # (Defunct) # spatial resolution
min_Rres = 0.01 # [kpc] <<< use 0.001 if want to resolve UCDs

#---stripping efficiency type
alpha_type = 'conc' # 'fixed' or 'conc'

#---dynamical friction strength
cfg.lnL_pref = 0.75 # Fiducial, but can also use 1.0

#---evolution mode (where subhaloes stop being evolved)
#
#   'fixed'     stop at a fixed halo mass, cfg.Mres.  Paper-1 behaviour.
#               Note this imposes a MASS-DEPENDENT effective f_b floor:
#               1e-3 for m_acc = 1e10 but 1e-1 for m_acc = 1e8, so
#               artificial disruption is worst for the lowest-mass
#               satellites.  Worth checking counts against 'arbres'.
#   'arbres'    stop at cfg.phi_res * m_acc.  Green+21 behaviour; only
#               defensible with the Green profile, since the
#               Penarrubia+10 tracks behind the Dekel profile are not
#               calibrated below f_b ~ 1e-3 (see CLAUDE.md issue 8).
#   'withering' stop at cfg.psi_res * M0.
#
# IMPORTANT: in 'fixed' mode keep lgMres_evo ~0.05 dex BELOW the tree's
# lgMres.  If they are equal, a subhalo accreted at exactly the tree
# resolution never gets a single msub call (see the lt sentinel below).
cfg.evo_mode = 'fixed' # 'fixed' | 'arbres' | 'withering'

lgMres_evo = 6.95      # [log10 Msun] used when evo_mode == 'fixed'
cfg.phi_res = 10**-5.0 # used when evo_mode == 'arbres'
cfg.psi_res = 10**-5.0 # used when evo_mode == 'withering'

# ev.msub takes the 'fixed' path iff cfg.Mres is not None, so the floor
# used inside msub and the disruption test below must be set together.
# Upstream SubEvo never set cfg.Mres, but this fork did while leaving
# min_mass at phi_res*m_acc -- the two disagreed and nothing ever
# terminated (CLAUDE.md issue 5).
if cfg.evo_mode == 'fixed':
    cfg.Mres = 10**lgMres_evo
else:
    cfg.Mres = None

########################### evolve satellites ###########################

tree_file_pattern = re.compile(r'tree\d+_lgM(\d+\.\d{3})\.npz')

def extract_host_mass(filepath):
    """
    Extract the host log-mass from a TreeGen output filename.
    """

    match = tree_file_pattern.search(os.path.basename(filepath))
    if match is None:
        return None
    return float(match.group(1))

def select_files_by_host_mass(files):
    """
    Optionally select a subset of tree files by host-mass bin or range.
    """

    file_masses = []
    for filepath in files:
        host_mass = extract_host_mass(filepath)
        if host_mass is not None:
            file_masses.append((filepath, host_mass))

    available_masses = sorted({host_mass for _, host_mass in file_masses})
    if not available_masses:
        print('>>> No valid tree files found in %s' % datadir, flush=True)
        return [], available_masses

    selected_mass = None
    selected_index = None
    mass_min = None
    mass_max = None

    if "SATGEN_HOST_MASS" in os.environ:
        selected_mass = float(os.environ["SATGEN_HOST_MASS"])
    elif "SATGEN_HOST_MASS_MIN" in os.environ or "SATGEN_HOST_MASS_MAX" in os.environ:
        if "SATGEN_HOST_MASS_MIN" in os.environ:
            mass_min = float(os.environ["SATGEN_HOST_MASS_MIN"])
        if "SATGEN_HOST_MASS_MAX" in os.environ:
            mass_max = float(os.environ["SATGEN_HOST_MASS_MAX"])
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        selected_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if selected_index < 0 or selected_index >= len(available_masses):
            raise IndexError(
                "SLURM_ARRAY_TASK_ID=%i is out of range for %i host-mass bins"
                % (selected_index, len(available_masses))
            )
        selected_mass = available_masses[selected_index]

    if selected_mass is not None:
        selected_files = [filepath for filepath, host_mass in file_masses
                          if host_mass == selected_mass]
        if selected_index is None:
            print('>>> Selected host mass bin: log(M)=%.3f' % selected_mass, flush=True)
        else:
            print('>>> SLURM_ARRAY_TASK_ID=%i -> host mass bin log(M)=%.3f'
                  % (selected_index, selected_mass), flush=True)
        return selected_files, available_masses

    if mass_min is not None or mass_max is not None:
        selected_files = []
        for filepath, host_mass in file_masses:
            if mass_min is not None and host_mass < mass_min:
                continue
            if mass_max is not None and host_mass > mass_max:
                continue
            selected_files.append(filepath)
        print('>>> Selected host mass range: %.3f to %.3f'
              % (mass_min if mass_min is not None else available_masses[0],
                 mass_max if mass_max is not None else available_masses[-1]),
              flush=True)
        return selected_files, available_masses

    print('>>> No host mass filter set; evolving all %i host-mass bins'
          % len(available_masses), flush=True)
    return [filepath for filepath, _ in file_masses], available_masses

#---get the list of data files
files = []    
for filename in os.listdir(datadir):
    if filename.startswith('tree') and filename.endswith('.npz'): 
        files.append(os.path.join(datadir, filename))
files.sort()
files, available_masses = select_files_by_host_mass(files)
host_mass_selected = np.unique([extract_host_mass(file) for file in files])
print("Host masses selected:", host_mass_selected)
if cfg.evo_mode == 'fixed':
    print('>>> evo_mode=fixed, cfg.Mres = 10^%.2f' % lgMres_evo, flush=True)
else:
    print('>>> evo_mode=%s, cfg.Mres = None (msub floors on the ratio)'
          % cfg.evo_mode, flush=True)

print('>>> Available host-mass bins: %s'
      % ', '.join('%.3f' % mass for mass in available_masses), flush=True)
print('>>> %d trees selected for evolution' % len(files), flush=True)
print('>>> Evolving subhaloes ...')

#---
time_start = time.time()
#for file in files: # <<< serial run, only for testing
def loop(file): 
    """
    Replaces the loop "for file in files:", for parallelization.
    """
    print(file)
    # skip if we already ran this one and are re-running
    # uncompleted trees on a second pass-through
    outfile = outdir + file[len(datadir):]
    label = os.path.basename(file)


    if(os.path.exists(outfile)):
        # NOTE: This will throw error if serial
        # Change the below to "continue" for serial
        print('    %s: output exists, skipping' % label, flush=True)
        return
        #continue

    time_start_tmp = time.time()
    print('    %s: starting' % label, flush=True)
    
    #---load trees
    f = np.load(file)
    redshift = f['redshift']
    CosmicTime = f['CosmicTime']
    mass = f['mass']
    order = f['order']
    ParentID = f['ParentID']
    VirialRadius = f['VirialRadius']
    concentration = f['concentration']
    coordinates = f['coordinates']

    # compute the virial overdensities for all redshifts
    VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL) # same as Dvsample
    GreenRte = np.zeros(VirialRadius.shape) - 99. # contains r_{te} values
    alphas = np.zeros(VirialRadius.shape) - 99.
    tdyns  = np.zeros(VirialRadius.shape) - 99.

    #---identify the roots of the branches
    izroot = mass.argmax(axis=1) # root-redshift ids of all the branches
    idx = np.arange(mass.shape[0]) # branch ids of all the branches
    levels = np.unique(order[order>=0]) # all >0 levels in the tree
    izmax = mass.shape[1] - 1 # highest redshift index

    #---get smallest host rvir from tree
    #   Defunct, we no longer use an Rres; all subhaloes are evolved
    #   until their mass falls below resolution limit
    min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0,:] > 0)[-1][0]]
    cfg.Rres = min(min_Rres, min_rvir * Rres_factor) # Never larger than 100 pc

    #---list of potentials and orbits for each branch
    #   additional, mass of ejected subhaloes stored in ejected_mass
    #   to be removed from corresponding host at next timestep
    potentials = [0] * mass.shape[0]
    orbits = [0] * mass.shape[0]
    trelease = np.zeros(mass.shape[0])
    ejected_mass = np.zeros(mass.shape[0])

    #---list of minimum masses, below which we stop evolving the halo
    M0 = mass[0,0]
    min_mass = np.zeros(mass.shape[0])

    total_steps = izmax

    #---evolve
    for iz in np.arange(izmax, 0, -1): # loop over time to evolve
        iznext = iz - 1                
        z = redshift[iz]
        tcurrent = CosmicTime[iz]
        tnext = CosmicTime[iznext]
        dt = tnext - tcurrent
        Dv = VirialOverdensity[iz]

        completed_steps = izmax - iz + 1
        if completed_steps == 1 or completed_steps % 50 == 0 or iz == 1:
            print('    %s: step %3i/%3i, z=%5.2f, active branches=%4i' % (
                label, completed_steps, total_steps, z, len(idx)
            ), flush=True)

        for level in levels: #loop from low-order to high-order systems
            for id in idx: # loop over branches
                if order[id,iz]!=level: continue # level by level
                if(iz <= izroot[id]):
                    if(iz == izroot[id]): # accretion happens at this timestep
                        # initialize Green profile and orbit

                        za = z
                        ta = tcurrent
                        Dva = Dv
                        ma = mass[id,iz] # initial mass that we will use for f_b
                        c2a = concentration[id,iz]
                        xva = coordinates[id,iz,:]

                        # some edge case produces nan in velocities in TreeGen
                        # if so, print warning and mass fraction lost
                        if(np.any(np.isnan(xva))):
                            print('    %s: WARNING: NaNs detected in init xv of id %d'\
                                % (label, id), flush=True)
                            print('    %s: mass fraction of tree lost: %.1e'\
                                % (label, ma/mass[0,0]), flush=True)
                            mass[id,:] = -99.
                            coordinates[id,:,:] = 0.
                            idx = np.delete(idx, np.argwhere(idx == id)[0])
                            # this is an extremely uncommon event, but should
                            # eventually be fixed
                            continue

                        potentials[id] = Green(ma,c2a,Delta=Dva,z=za)
                        orbits[id] = orbit(xva)
                        trelease[id] = ta

                        if cfg.evo_mode == 'fixed':
                            min_mass[id] = cfg.Mres
                        elif cfg.evo_mode == 'arbres':
                            min_mass[id] = cfg.phi_res * ma
                        elif cfg.evo_mode == 'withering':
                            min_mass[id] = cfg.psi_res * M0
                        else:
                            raise ValueError('bad evo_mode: %s'
                                             % cfg.evo_mode)

                    #---main loop for evolution

                    # the p,s,o objects are updated in-place in their arrays
                    # unless the orbit is replaced with a new object when released
                    ip = ParentID[id,iz]
                    p = potentials[ip]
                    s = potentials[id]

                    # lt and rte are function locals that persist across
                    # the whole loop(file) call.  In SatEvo's
                    # branch-outer ordering a stale value was this
                    # branch's own previous timestep, which was the
                    # documented intent.  In this z-outer ordering it
                    # would be ANOTHER BRANCH's value, and VirialRadius
                    # feeds the high-order release test.  Sentinel them.
                    lt = None
                    rte = None

                    # update mass of subhalo object based on mass-loss in previous snapshot
                    # we wait to do it until now so that the pre-stripped subhalo can be used
                    # in the evolution of any higher-order subhaloes
                    # We also strip off the mass of any ejected systems
                    # the update_mass function handles cases where we fall below resolution limit
                    if(s.Mh > min_mass[id]):
                        if(ejected_mass[id] > 0):
                            mass[id,iz] -= ejected_mass[id]
                            ejected_mass[id] = 0
                            mass[id,iz] = max(mass[id,iz], min_mass[id])

                        s.update_mass(mass[id,iz])
                        rte = s.rte()

                    o = orbits[id]
                    xv = orbits[id].xv
                    m = s.Mh
                    m_old = m
                    r = np.sqrt(xv[0]**2+xv[2]**2)

                    #---time since in current host
                    t = tnext - trelease[id]

                    # Order should always be one higher than parent unless 
                    # ejected,in which case it should be the same as parent
                    k = order[ip,iznext] + 1

                    # alpha: stripping efficiency
                    if(alpha_type == 'fixed'):
                        alpha = 0.55
                    elif(alpha_type == 'conc'):
                        alpha = ev.alpha_from_c2(p.ch, s.ch)

                    #---evolve satellite
                    # as long as the mass is larger than resolution limit
                    if m > min_mass[id]:

                        # evolve subhalo properties
                        m,lt = ev.msub(s,p,xv,dt,choice='King62',
                            alpha=alpha)

                    else: # we do nothing about disrupted satellite, s.t.,
                        # its properties right before disruption would be 
                        # stored in the output arrays
                        pass

                    #---evolve orbit
                    if m > min_mass[id]:
                        # NOTE: We previously had an additional check on r>Rres
                        # here, where Rres = 10^-3 Rvir(z), but I removed it
                        # All subhalo orbits are evolved until their mass falls
                        # below the resolution limit.
                        # NOTE: No use integrating orbit any longer once the halo
                        # is disrupted, this just slows it down
                    
                        tdyn = p.tdyn(r)
                        o.integrate(t,p,m_old)
                        xv = o.xv # note that the coordinates are updated 
                        # internally in the orbit instance "o" when calling
                        # the ".integrate" method, here we assign them to 
                        # a new variable "xv" only for bookkeeping
                        
                    else: # i.e., the satellite has merged to its host, so
                        # no need for orbit integration; to avoid potential 
                        # numerical issues, we assign a dummy coordinate that 
                        # is almost zero but not exactly zero
                        tdyn = p.tdyn(cfg.Rres)
                        xv = np.array([cfg.Rres,0.,0.,0.,0.,0.])

                    r = np.sqrt(xv[0]**2+xv[2]**2)
                    m_old = m


                    #---if order>1, determine if releasing this high-order 
                    #   subhalo to its grandparent-host, and if releasing,
                    #   update the orbit instance
                    if k>1:
                    
                        if (r > VirialRadius[ip,iz]) & (iz <= izroot[ip]): 
                            # <<< Release condition:
                            # 1. Host halo is already within a grandparent-host
                            # 2. Instant orbital radius is larger than the host
                            # TIDAL radius (note that VirialRadius also contains
                            # the tidal radii for the host haloes once they fall
                            # into a grandparent-host)
                            # 3. (below) We compute the fraction of:
                            #             dynamical time / alpha
                            # corresponding to this dt, and release with
                            # probability dt / (dynamical time / alpha)

                            # Compute probability of being ejected
                            odds = np.random.rand()
                            dyntime_frac = alphas[ip,iz] * dt / tdyns[ip,iz]
                            if(odds < dyntime_frac):
                                if(ParentID[ip,iz] == ParentID[ip,iznext]):
                                    # host wasn't also released at same time
                                    # New coordinates at next time are the
                                    # updated subhalo coordinates plus the updated
                                    # host coordinates inside of grandparent
                                    xv = aux.add_cyl_vecs(xv,coordinates[ip,iznext,:])
                                else:
                                    xv = aux.add_cyl_vecs(xv,coordinates[ip,iz,:])
                                    # This will be extraordinarily rare, but just
                                    # a check in case so that the released order-k
                                    # subhalo isn't accidentally double-released
                                    # in terms of updated coordinates, but not
                                    # in terms of new host ID.
                                orbits[id] = orbit(xv) # update orbit object
                                k = order[ip,iz] # update instant order to the same as the parent
                                ejected_mass[ip] += m 
                                # add updated subhalo mass to a bucket to be removed from host
                                # at start of next timestep
                                ip = ParentID[ip,iz] # update parent id
                                trelease[id] = tnext # update release time

                    #---update the arrays for output
                    mass[id,iznext] = m
                    order[id,iznext] = k
                    ParentID[id,iznext] = ip
                    if lt is not None:
                        # NOTE: We store tidal radius in lieu of virial
                        # radius for haloes after they start getting
                        # stripped
                        VirialRadius[id,iznext] = lt
                    else:
                        # Subhalo is at/below the resolution floor and was
                        # not evolved this step, so it has no new tidal
                        # radius.  Freeze its own last value rather than
                        # inheriting another branch's (upstream raised
                        # UnboundLocalError here and aborted the whole
                        # tree file, writing no output at all).
                        VirialRadius[id,iznext] = VirialRadius[id,iz]

                    if rte is not None:
                        GreenRte[id,iz] = rte
                        # left at -99 otherwise, which is what the output
                        # comment below promises
                    coordinates[id,iznext,:] = xv

                    # NOTE: the below two are quantities at current timestep
                    # instead, since only used for host release criteria
                    # This won't be output since only used internally
                    alphas[id,iz] = alpha
                    tdyns[id,iz] = tdyn

                else: # before accretion, halo is an NFW profile
                    if(concentration[id,iz] > 0): 
                        # the halo has gone above tree mass resolution
                        # different than SatEvo mass resolution by small delta
                        potentials[id] = NFW(mass[id,iz],concentration[id,iz],
                                             Delta=VirialOverdensity[iz],z=redshift[iz])

    #---output
    np.savez(outfile, 
        redshift = redshift,
        CosmicTime = CosmicTime,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        GreenRte = GreenRte,
        # this contains values during stripping, -99 prior to stripping and
        # once the halo falls below the resolution limit
        concentration = concentration, # this is unchanged from TreeGen output
        coordinates = coordinates,
        )
    print('    %s: output saved' % label, flush=True)

    #---on-screen prints
    m0 = mass[:,0][1:]
    
    msk = (m0 > min_mass[1:]) & (m0 < M0) & (order[1:,0] == 1)
    fsub = m0[msk].sum() / M0
    
    MAH = mass[0,:]
    iz50 = aux.FindNearestIndex(MAH,0.5*M0)
    z50 = redshift[iz50]
    
    time_end_tmp = time.time()
    print('    %s: %5.2f min, z50=%5.2f,fsub=%8.5f'%\
        (outfile,(time_end_tmp-time_start_tmp)/60., z50,fsub), flush=True)

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    if len(sys.argv) > 1:
        Ncores = int(sys.argv[1])
    else:
        Ncores = cpu_count()
    print('    using %i cores' % Ncores, flush=True)
    pool = Pool(Ncores) # use as many as requested
    pool.map(loop, np.random.permutation(files), chunksize=1)

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.), flush=True)

# python SubEvo.py 1
