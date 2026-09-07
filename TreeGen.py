################################ TreeGen ################################

# Generate halo merger trees using the Parkinson et al. (2008) algorithm,
# and initialize haloes with Dekel+17 profiles that include the baryonic
# halo response of Tollet+16 ('NIHAO') or Bose+19 ('APOSTLE').

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016 Hebrew University
# Arthur Fangzhou Jiang 2019 Hebrew University
# Sheridan Beckwith Green 2020 Yale University

# Jiaxuan Li -- ELVES-Dwarf 'hybrid' branch.
# This is the Dekel + baryonic-response tree generator; TreeGen_Sub.py
# remains the DMO/NFW counterpart.  Both now write BN98 virial
# quantities on the same redshift grid, so a tree from either file can
# be evolved by the same driver.
#
# Changes relative to the upstream TreeGen.py:
#   1. Virial quantities use the Bryan & Norman (1998) overdensity
#      (cfg.Dvsample) instead of a hard-coded Delta = 200 rho_crit.
#      Required for consistency with the Li+20 orbit sampler (which
#      assumes BN98 by its own docstring), with the Jiang+19 R_eff
#      relation, and with the DASH-calibrated Green model.
#   2. ParentID is int32.  int16 silently overflows above 32767
#      branches, which happens for lgM0 >~ 12 at lgMres = 7.
#   3. Fixed the "len(c)==0" safety branch: upstream unpacked 4 return
#      values from init.Dekel_fromMAH (which returns 5) and appended an
#      undefined Rvi.  Both would raise.
#   4. Guard for the case where aux.downsample returns scalars.
#   5. Selectable infall-orbit sampler: 'zzli' (Li+20), 'jiang'
#      (Jiang+15), or 'zentner' (Zentner+05).
#   6. Also stores DMOconcentration, i.e. c_-2 BEFORE the baryonic
#      response, so that the same tree file can drive a DMO/NFW/Green
#      evolution run.  This makes the Dekel-vs-Green comparison paired
#      (identical trees and identical infall orbits) rather than merely
#      statistical.
#   7. SLURM-aware driver with retries and %.3f in the filename, so the
#      output matches what SubEvo.py's filename parser expects.
#
# NOTE: init.Dekel_fromMAH redraws the 0.2 dex scatter on M_star and the
# 0.1 scatter on c_-2/c_-2,DMO independently at every output redshift,
# so the Dekel parameters along a branch carry uncorrelated snapshot
# noise.  That is upstream behaviour, kept here deliberately; changing
# it would change results and should be a separate decision.

######################## set up the environment #########################

#---user modules
import config as cfg
import cosmo as co
import init
from profiles import Dekel, NFW
import aux

#---python modules
import numpy as np
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from os import path

# <<< for clean on-screen prints, use with caution, make sure that
# the warning is not prevalent or essential for the result
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

############################# user control ##############################

overwrite = True # if False, trees with existing output files are skipped

#---target halo, desired resolution
idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
print('>>> SLURM_ARRAY_TASK_ID: %i' % idx, flush=True)

hmasses = np.arange(10.5, 12.6, 0.01) # same grid as TreeGen_Sub.py
lgM0 = hmasses[idx]

z0 = 0.
lgMres = 7.0

#---orbital parameter sampler preference
optype = 'zzli' # 'zzli' or 'zentner' or 'jiang'

#---baryonic-effect choice
HaloResponse = 'NIHAO' # 'NIHAO' (Tollet+16) or 'APOSTLE' (Bose+19)

#---optional guard on the Dekel inner slope
# init.aDekel inverts the target slope s_0.01 into the Dekel alpha via
#     alpha = (s + 2u) / (1 + u),    u = (s - 3.5) sqrt(c_-2) / 15
# The numerator turns negative whenever s < 7 sqrt(c_-2)/(15 + 2 sqrt(c_-2)),
# and the denominator has a pole at c_-2 = (15/(3.5-s))^2 (c_-2 ~ 18-25 for
# a strongly cored halo).  Upstream applies no guard, so alpha can come out
# negative -- a central density HOLE rather than a core -- or blow up near
# the pole.  In the paper-1 trees this hits ~1% of all entries but 63% of
# entries at z < 0.05, i.e. almost entirely the late-time host and the
# recently accreted satellites.
# alpha_range = None  reproduces upstream behaviour exactly.
# alpha_range = (0., 1.9)  clips to a physical core-to-cusp range and
#                          recomputes the Dekel c consistently.
alpha_range = None

#---for output
outdir = '/scratch/gpfs/JENNYG/jiaxuanl/SatGen/OUTPUT_TREE_DEKEL_%s/' % HaloResponse
outfile1 = outdir + 'tree%i_lgM%.3f.npz' #%(itree,lgM0)

max_retries = int(os.environ.get("SATGEN_MAX_RETRIES", 0))

print('****************************************')
print('log halo mass: %.3f' % lgM0)
print('halo response: %s' % HaloResponse)
print('orbit sampler: %s' % optype)
print(outfile1)
print('****************************************', flush=True)

############################### compute #################################

def _clip_dekel(ci, ai, c2i):
    """
    Optionally clip the Dekel inner slope into alpha_range, recomputing
    the Dekel concentration so that (c, alpha, c_-2) stay consistent.
    Returns the pair unchanged if alpha_range is None.
    """
    if alpha_range is None:
        return ci, ai
    ai_clipped = min(max(ai, alpha_range[0]), alpha_range[1])
    if ai_clipped == ai:
        return ci, ai
    return init.cDekel(c2i, ai_clipped), ai_clipped

time_start = time.time()

def _generate_tree(itree, attempt):
    """
    Generate a single merger tree.
    """

    time_start_tmp = time.time()
    print('    Tree %5i: starting attempt %i/%i' % (
        itree, attempt, max_retries + 1
    ), flush=True)

    np.random.seed() # [important!] reseed the random number generator

    cfg.M0 = 10.**lgM0
    cfg.z0 = z0
    cfg.Mres = 10.**lgMres
    cfg.Mmin = 0.04*cfg.Mres

    k = 0               # the level, k, of the branch being considered
    ik = 0              # how many level-k branches have been finished
    Nk = 1              # total number of level-k branches
    Nbranch = 1         # total number of branches in the current tree

    Mak = [cfg.M0]      # accretion masses of level-k branches
    zak = [cfg.z0]
    idk = [0]           # branch ids of level-k branches
    ipk = [-1]          # parent ids of level-k branches (-1: no parent)

    Mak_tmp = []
    zak_tmp = []
    idk_tmp = []
    ipk_tmp = []

    mass = np.zeros((cfg.Nmax,cfg.Nz)) - 99.
    order = np.zeros((cfg.Nmax,cfg.Nz),np.int8) - 99
    ParentID = np.zeros((cfg.Nmax,cfg.Nz),np.int32) - 99

    VirialRadius = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    concentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    DMOconcentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    DekelConcentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    DekelSlope = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.

    StellarMass = np.zeros((cfg.Nmax,cfg.Nz)) - 99.
    StellarSize = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.

    coordinates = np.zeros((cfg.Nmax,cfg.Nz,6),np.float32)

    while True: # loop over branches, until the full tree is completed.
    # Starting from the main branch, draw progenitor(s) using the
    # Parkinson+08 algorithm. When there are two progenitors, the less
    # massive one is the root of a new branch. We draw branches level by
    # level, i.e., When a new branch occurs, we record its root, but keep
    # finishing the current branch and all the branches of the same level
    # as the current branch, before moving on to the next-level branches.

        M = [Mak[ik]]   # mass history of current branch in fine timestep
        z = [zak[ik]]   # the redshifts of the mass history
        cfg.M0 = Mak[ik]# descendent mass
        cfg.z0 = zak[ik]# descendent redshift
        id = idk[ik]    # branch id
        ip = ipk[ik]    # parent id

        while cfg.M0>cfg.Mmin:

            if cfg.M0>cfg.Mres: zleaf = cfg.z0 # update leaf redshift

            co.UpdateGlobalVariables(**cfg.cosmo)
            M1,M2,Np = co.DrawProgenitors(**cfg.cosmo)

            # update descendent halo mass and descendent redshift
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0+cfg.dW)
            if cfg.z0>cfg.zmax: break

            if Np>1 and cfg.M0>cfg.Mres: # register next-level branches

                Mak_tmp.append(M2)
                zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch)
                ipk_tmp.append(id)
                Nbranch += 1

            # record the mass history at the original time resolution
            M.append(cfg.M0)
            z.append(cfg.z0)

        # Now that a branch is fully grown, do some book-keeping

        # convert mass-history list to array
        M = np.array(M)
        z = np.array(z)

        # downsample the fine-step mass history, M(z), onto the
        # coarser output timesteps, cfg.zsample
        Msample,zsample = aux.downsample(M,z,cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample,zsample)
        if np.isscalar(iz) or (np.ndim(iz) == 0):
            iz = np.array([iz]) # avoids error in loop below
            zsample = np.atleast_1d(zsample)
            Msample = np.atleast_1d(Msample)
        izleaf = aux.FindNearestIndex(cfg.zsample,zleaf)
        # Note: zsample[j] is same as cfg.zsample[iz[j]]

        # compute halo structure throughout time on the coarse grid, up
        # to the leaf point
        t = co.t(z,cfg.h,cfg.Om,cfg.OL)
        c,a,c2,c2DMO,Rv = [],[],[],[],[]
        for i in iz:
            if i > (izleaf+1): break # only compute structure below leaf
            msk = z>=cfg.zsample[i]
            if True not in msk: break # safety
            ci,ai,Msi,c2i,c2DMOi = init.Dekel_fromMAH(M[msk],t[msk],
                cfg.zsample[i],HaloResponse=HaloResponse)
            ci,ai = _clip_dekel(ci,ai,c2i)
            Rvi = init.Rvir(M[msk][0],Delta=cfg.Dvsample[i],z=cfg.zsample[i])
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            c2DMO.append(c2DMOi)
            Rv.append(Rvi)
            if i==iz[0]: Ms = Msi
            #print('    i=%6i,ci=%8.2f,ai=%8.2f,log(Msi)=%8.2f,c2i=%8.2f'%\
            #    (i,ci,ai,np.log10(Msi),c2i)) # <<< for test
        if len(c)==0: # <<< safety, dealing with rare cases where the
            # branch's root z[0] is close to the maximum redshift -- when
            # this happens, the mass history has only one element, and
            # z[0] can be slightly above cfg.zsample[i] for the very
            # first iteration, leaving the lists c,a,c2,Rv never updated
            ci,ai,Msi,c2i,c2DMOi = init.Dekel_fromMAH(M,t,z[0],
                HaloResponse=HaloResponse)
            ci,ai = _clip_dekel(ci,ai,c2i)
            Rvi = init.Rvir(M[0],Delta=cfg.Dvsample[iz[0]],
                z=cfg.zsample[iz[0]])
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            c2DMO.append(c2DMOi)
            Rv.append(Rvi)
            Ms = Msi
        c = np.array(c)
        a = np.array(a)
        c2 = np.array(c2)
        c2DMO = np.array(c2DMO)
        Rv = np.array(Rv)
        Nc = len(c2) # length of a branch over which c2 is computed

        # compute stellar size at the root of the branch, i.e., at the
        # accretion epoch (z[0])
        Re = init.Reff(Rv[0],c2[0])

        # use the redshift id and parent-branch id to access the parent
        # branch's information at our current branch's accretion epoch,
        # in order to initialize the orbit
        if ip==-1: # i.e., if the branch is the main branch
            xv = np.zeros(6)
        else:
            Mp = mass[ip,iz[0]]
            cp = DekelConcentration[ip,iz[0]]
            ap = DekelSlope[ip,iz[0]]
            hp = Dekel(Mp,cp,ap,Delta=cfg.Dvsample[iz[0]],z=zsample[0])

            if(optype == 'zentner'):
                eps = 1./np.pi*np.arccos(1.-2.*np.random.random())
                xv = init.orbit(hp,xc=1.,eps=eps)
            elif(optype == 'zzli'):
                vel_ratio, gamma = init.ZZLi2020(hp, Msample[0], zsample[0])
                xv = init.orbit_from_Li2020(hp, vel_ratio, gamma)
            elif(optype == 'jiang'):
                sp = NFW(Msample[0],c2[0],Delta=cfg.Dvsample[iz[0]],
                    z=zsample[0])
                xv = init.orbit_from_Jiang2015(hp,sp,zsample[0])
            else:
                sys.exit('Invalid optype: %s' % optype)

        # update the arrays for output
        mass[id,iz] = Msample
        order[id,iz] = k
        ParentID[id,iz] = ip

        VirialRadius[id,iz[0]:iz[0]+Nc] = Rv
        concentration[id,iz[0]:iz[0]+Nc] = c2
        DMOconcentration[id,iz[0]:iz[0]+Nc] = c2DMO
        DekelConcentration[id,iz[0]:iz[0]+Nc] = c
        DekelSlope[id,iz[0]:iz[0]+Nc] = a

        StellarMass[id,iz[0]] = Ms
        StellarSize[id,iz[0]] = Re

        coordinates[id,iz[0],:] = xv

        # Check if all the level-k branches have been dealt with: if so,
        # i.e., if ik==Nk, proceed to the next level.
        ik += 1
        if ik==Nk: # all level-k branches are done!
            Mak = Mak_tmp
            zak = zak_tmp
            idk = idk_tmp
            ipk = ipk_tmp
            Nk = len(Mak)
            ik = 0
            Mak_tmp = []
            zak_tmp = []
            idk_tmp = []
            ipk_tmp = []
            if Nk==0:
                break # jump out of "while True" if no next-level branch
            k += 1 # update level

    # trim and output
    mass = mass[:id+1,:]
    order = order[:id+1,:]
    ParentID = ParentID[:id+1,:]
    VirialRadius = VirialRadius[:id+1,:]
    concentration = concentration[:id+1,:]
    DMOconcentration = DMOconcentration[:id+1,:]
    DekelConcentration = DekelConcentration[:id+1,:]
    DekelSlope = DekelSlope[:id+1,:]
    StellarMass = StellarMass[:id+1,:]
    StellarSize = StellarSize[:id+1,:]
    coordinates = coordinates[:id+1,:,:]
    np.savez(outfile1%(itree,lgM0),
        redshift = cfg.zsample,
        CosmicTime = cfg.tsample,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        concentration = concentration,
        DMOconcentration = DMOconcentration,
        DekelConcentration = DekelConcentration,
        DekelSlope = DekelSlope,
        StellarMass = StellarMass,
        StellarSize = StellarSize,
        coordinates = coordinates,
        HaloResponse = HaloResponse,
        )

    time_end_tmp = time.time()
    print('    Tree %5i: log(M_0)=%6.3f, %6i branches, %2i order, %8.1f sec'\
        %(itree,lgM0,Nbranch,k,time_end_tmp-time_start_tmp), flush=True)

def loop(itree):
    """
    Replaces the loop "for itree in range(Ntree):", for parallelization.
    """

    outfile = outfile1 % (itree, lgM0)
    if not overwrite and path.exists(outfile):
        print('    Tree %5i: output exists, skipping' % itree, flush=True)
        return

    for attempt in range(1, max_retries + 2):
        try:
            _generate_tree(itree, attempt)
            return
        except Exception as exc:
            if path.exists(outfile):
                os.remove(outfile)
            if attempt > max_retries:
                print('    Tree %5i: failed after %i attempts: %s: %s' % (
                    itree, attempt, type(exc).__name__, exc
                ), flush=True)
                raise
            print('    Tree %5i: attempt %i failed with %s: %s; retrying' % (
                itree, attempt, type(exc).__name__, exc
            ), flush=True)

if __name__ == "__main__":
    Ntree = int(sys.argv[1])
    os.makedirs(outdir, exist_ok=True)
    print('>>> CPU count: %i' % cpu_count(), flush=True)
    ncores = int(os.environ.get("SLURM_CPUS_PER_TASK", cpu_count()))
    ncores = max(1, min(ncores, Ntree))
    print('    using %i cores' % ncores, flush=True)
    print('>>> Generating %i trees for log(M_0)=%.3f at log(M_res)=%.3f...'%\
        (Ntree, lgM0, lgMres), flush=True)
    pool = Pool(processes=ncores, maxtasksperchild=1)
    try:
        pool.map(loop, range(Ntree), chunksize=1)
    except BaseException:
        pool.terminate()
        raise
    else:
        pool.close()
    finally:
        pool.join()

time_end = time.time()
print('    total time: %5.2f hours'%((time_end - time_start)/3600.), flush=True)

# python TreeGen.py 1
