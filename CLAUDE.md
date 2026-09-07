# SatGen — ELVES-Dwarf hybrid pipeline

Working notes for the `hybrid` branch. Read this before touching the
evolution or tree code.

## Hard requirement: run from the repo root

`config.py` loads its interpolation tables with **relative** paths
(`np.load('etc/gvdb_mm.npy')`), and all modules use flat imports
(`import config as cfg`). Everything must be run with
`/home/jiaxuanl/Research/SatGen` as the working directory.

`evolve.g_EPW18` uses `scipy.interpolate.interp2d`, removed in scipy
>= 1.14. Currently on scipy 1.11. Port to `RegularGridInterpolator`
before upgrading, or keep the pin.

## Branches

| Branch | What it is |
|---|---|
| `master` | Jiang+21 SatGen (arXiv 2005.05974), packaged as a `SatGen/` module by aphearin and reformatted with black. This is what `ELVES-Dwarf/script/SatGen_script/run_satgen/` imports. Paper 1. |
| `shergreen-master` | Green+21 (arXiv 2110.13044) upstream + Jiaxuan's SLURM work. Paper 2. |
| `hybrid` | **Current work.** Branched from `shergreen-master` @ `f8d2139`. |

The two upstream branches diverged at `3b04fd7`. A token-level diff
(normalising whitespace, comments, docstrings, float literals) shows the
**physics library is identical** between them — every semantic
difference is an import statement, except:

- `aux.downsample`: `xgrid[idx1+1]` (master, can IndexError) vs
  `xgrid[idx1]` (shergreen-master, correct).
- `config.zsample`: master overshoots `zmax` by one snapshot and appends
  a trailing `dt=0`. Benign; alignment is the same in both.

So "old vs new SatGen" is **not** two codebases. It is two driver
scripts selecting different options from one library.

## The two pipelines

|  | old: `TreeGen.py` + `SatEvo.py` | new: `TreeGen_Sub.py` + `SubEvo.py` |
|---|---|---|
| Subhalo profile | Dekel+17, evolved on Penarrubia+10 tidal tracks (`ev.Dekel2`) | NFW x DASH transfer function (`profiles.Green`) |
| Baryonic halo response | Tollet+16 `NIHAO` / Bose+19 `APOSTLE` | none (DMO) |
| Galaxies | M*, R_e evolved in-loop via `ev.g_EPW18` | none; painted post hoc |
| Host disk | optional `MN` (`fd`, `fb`); **`fd=0` in the paper-1 fiducial** | none |
| Virial definition | `Delta=200` rho_crit | BN98 (`cfg.Dvsample`) |
| Stripping efficiency | fixed `alpha=0.6` | `ev.alpha_from_c2` (DASH fit) |
| Loop order | branch-outer / time-inner | **z-outer / branch-inner** |
| Mass conservation | no | yes (`ejected_mass[ip]`) |
| High-order release | deterministic `r > R_vir` | probabilistic `P = alpha*dt/t_dyn` |
| Resolution | fixed `cfg.Mres` | `arbres`: `phi_res * m_acc` |

## Hybrid design decisions

| Aspect | Choice |
|---|---|
| Merger tree | Parkinson+08, unchanged |
| Virial definition | **BN98 everywhere.** Required by `init.orbit_from_Li2020` (says so in its docstring), by `gh.Reff` (Jiang+19 is a virial relation), and by DASH. |
| Host profile | Dekel + response on the main branch; NFW for order >= 1 hosts (no response prescription exists for a stripped host) |
| Host disk | MN, `k == 1` only |
| Subhalo profile | **Runtime switch, not a merge.** Green fiducial; Dekel/NIHAO and Dekel/APOSTLE as the systematic bracket. Quote the spread. |
| Stripping efficiency | `ev.alpha_from_c2`, fed **c_-2** (see gotcha 4) |
| Mass resolution | Fixed `cfg.Mres` (Jiaxuan's choice). Set the evolution floor ~0.05 dex **below** the tree floor (see gotcha 6). |
| Loop order | z-outermost — keep `SubEvo`'s |
| Mass conservation | keep |
| High-order release | probabilistic — keep |
| DF | `lnL_pref = 0.75` |
| Orbit sampler | Li+20 (`optype='zzli'`) |
| Galaxies | in-loop `g_EPW18` if tidally-stripped M* is wanted; needs per-branch state arrays |

**Start the evolution code from `SubEvo.py`, not `SatEvo.py`.** All of
the structural work (z-outer loop, mass conservation, probabilistic
release, BN98, int32) already exists there. Going the other direction
means converting ~15 per-branch locals into arrays and rediscovering the
deferred `update_mass` ordering.

## Progress

- [x] `hybrid` branch created from `shergreen-master` @ `f8d2139`
- [x] **`TreeGen.py` rewritten** (commit `c5171cd`) — Dekel + response
      tree generator, BN98 throughout, int32 `ParentID`, fixed the
      `len(c)==0` safety branch (it unpacked 4 of 5 return values and
      appended an undefined `Rvi`), scalar-`iz` guard, orbit-sampler
      switch, SLURM + retries, `%.3f` filename, new `DMOconcentration`
      output, optional `alpha_range` guard.
      Verified: R_vir/R_200c = 1.2552 at z=0 (Delta = 101.1); 0 empty
      branch rows; `ParentID` int32.
- [ ] `SubEvo.py` fork: fixed-`Mres` switch + stale-`lt` fix first, then
      the Dekel branch, then galaxies, then the disk
- [ ] `TreeGen_Sub.py` left as the DMO path (unchanged)

## Open decision

**`alpha_range` in `TreeGen.py`.** Default `None` reproduces upstream
exactly. `(0., 1.9)` clips the Dekel inner slope and recomputes the
Dekel c consistently. Measured on a smoke-test tree:

```
alpha_range=None    : frac a<0 = 0.177   a in [-44.84, 368.75]   cDekel max = 1.7e6
alpha_range=(0,1.9) : frac a<0 = 0.000   a in [  0.00,   1.90]   cDekel max = 74
```

Recommendation: **`(0., 1.9)`**, upgraded from "your call" once the
dwarf-run numbers came in -- 96.4% of low-z host snapshots in
`OUTPUT_SAT_DWARF_..._ZZLi_11` have alpha < 0, with negative and NaN
densities following. Changes results relative to paper 1, so it is still
Jiaxuan's call. **Not yet decided.**

## Known issues (all verified against real data)

1. **`init.aDekel` has a pole; alpha goes negative.**
   `alpha = (s + 2u)/(1 + u)` with `u = (s-3.5)*sqrt(c_-2)/15`.
   The numerator turns negative when `s < 7*sqrt(c2)/(15 + 2*sqrt(c2))`;
   the denominator has a pole at `c_-2 = (15/(3.5-s))^2` (~18-25 for a
   cored halo). Upstream applies no guard, so alpha < 0 means a central
   density *hole*, not a core.
   In `sdanieli/OUTPUT_SAT_fd0.00_fb0.00_NIHAO_0/tree0_lgM13.49.npz`:
   1.1% of all entries, rising to 1.6% at z < 0.5; `cDekel` max 2.8e8;
   the main host at z=0 has alpha = -1.69.

   **Far worse in the dwarf runs.** In
   `OUTPUT_SAT_DWARF_fd0.00_fb0.00_NIHAO_ZZLi_11`, over 30 hosts and
   snapshots iz < 50 (z < 0.26):

   ```
   alpha < 0               : 0.964   <- 96% of low-z host snapshots
   rho(0.1 Rv) NaN         : 0.009
   rho(0.1 Rv) NEGATIVE    : 0.015
   M(<0.1 Rv) > M_host     : 0.019   <- unphysical
   ```

   The host parameters do not drift, they thrash between adjacent
   snapshots (see issue 9), e.g. for `tree11_lgM11.50.npz`:
   alpha = -374 (iz=0), -29 (iz=1), **+41** (iz=2), -6.9 (iz=3),
   **+12.5** (iz=4). At iz=2 and iz=4, `rho` is negative and
   M(<10 kpc) = 1.5e12 against a host mass of 3.2e11.

   NaN mechanism: `x**alpha` underflows to 0 while
   `(1+sqrt(x))**(2*(3.5-alpha))` overflows to inf, so `0*inf = nan`.

   **This is probably why the scipy pin exists.** Reproducing the z=0
   stripping step for that file raises
   `ValueError: The function value at x=0.01 is NaN` from scipy's
   `brentq` NaN guard, yet the shipped output has no NaNs and a normal
   z=0 step. Most likely scipy 1.10.1 lacked that guard and `brentq`
   returned a garbage root instead of raising -- which matches
   run_satgen/README.md ("Newer versions of scipy will make SatGen
   crash"). If so the pin masked this bug rather than avoiding it.
   NOT yet proven; confirming means installing scipy 1.10.1 and
   re-running one satellite.

   Affects every paper-1 run. Mitigated by `alpha_range`; given the 96%
   figure, **recommend turning it on**.

2. **`ev.lt_King62_RHS` evaluates the host density in the midplane.**
   It calls `pr.rho(potential, r)`, and `pr.rho(potential, R, z=0.)`
   defaults z to 0. Correct for spherical components; badly wrong for an
   MN disk, which gets evaluated at maximum midplane density regardless
   of the satellite's z. `dlnM/dlnr` is inflated, the RHS flips
   negative, no root is bracketed, and `ltidal` **silently returns
   `cfg.Rres`** — so `msub` strips everything outside 1 pc every step.
   `lt_Tormen98_RHS` has the same bug and is worse (no centrifugal
   term). `pr.M` is fine — `MN.M` is a spherical enclosed mass.
   At the real `sdanieli` fd=0.1 parameters: **77.2%** of order-1
   satellites collapse to `Rres`, vs **5.8%** with
   `rho = pr.rho(potential, xv[0], xv[2])`. Observable effect on that
   run: f_sub 0.200 (fd=0) -> 0.115 (fd=0.1).
   The residual 5.8% are genuine midplane crossings where King62 has no
   solution; those need a fallback (hold previous `lt`), not `Rres`.
   **Any run with `fd > 0` is affected** — including
   `sdanieli/OUTPUT_SAT_fd0.10_fb0.00_NIHAO_0`.
   **Runs with `fd = 0` are NOT affected** -- the `if (fd > 0.0) and
   (k == 1)` gate in `SatEvo.py` leaves the potential as a single
   spherical `Dekel`, for which `pr.rho(potential, r)` is correct.
   Checked explicitly on
   `OUTPUT_SAT_DWARF_fd0.00_fb0.00_NIHAO_ZZLi_11`: the King62 RHS was
   never negative at iz = 0, 5, 10, 20, 40, 70, 100
   (median dlnM/dlnr ~ 0.3-0.6).

3. **`Dekel` has no `.Minit`.** `ev.msub`'s arbres branch does
   `max(sp.Mh-dm, cfg.phi_res*sp.Minit)`, which only `Green` provides.
   `AttributeError` if Dekel is used with `cfg.Mres = None`. One-line
   fix (`self.Minit = M` in `Dekel.__init__`).

4. **`ev.alpha_from_c2` wants c_-2, and `Dekel.ch` is not c_-2.**
   For `NFW`/`Green`, `.ch` is c_-2. For `Dekel`, `.ch` is the Dekel c,
   with `c_-2 = 2.25*c_Dekel/(2-alpha)^2` (ratio 0.56 to 2.25 over
   alpha in [0,1]). Since alpha ~ c_s^(-1/3), using `.ch` misestimates
   the stripping efficiency by up to 31%. Add a `.c2()` accessor to each
   profile class rather than reading `.ch`.

5. **`SubEvo.py` sets `cfg.Mres = 10**7.0` (line ~152) while
   `min_mass[id] = cfg.phi_res * ma`.** These disagree: `ev.msub` floors
   mass at 1e7 but the disruption test is at `1e-5*m_acc`, so nothing
   ever terminates. Upstream never sets `cfg.Mres`. Must be made
   consistent in the hybrid. Probably explains the ~340 min/tree runtime.

6. **Do not set the evolution floor equal to the tree floor.** A subhalo
   accreted at exactly the tree resolution never gets an `msub` call,
   `lt` is unbound, and the `except UnboundLocalError` handler does
   `return` — **silently aborting the whole tree file with no output**.
   Use tree `lgMres = 7.0`, evolution `cfg.Mres = 10**6.95`.

7. **Stale `lt` / `rte` across branches in `SubEvo.py`.** Both are
   function locals that persist for the whole `loop(file)` call. When a
   subhalo is below the floor neither is recomputed, yet both are still
   written to the output arrays. In `SatEvo`'s branch-outer ordering the
   stale value is that branch's own previous timestep (intended
   behaviour); in `SubEvo`'s z-outer ordering it comes from **a
   different branch**. `VirialRadius[ip,iz]` feeds the release test, so
   this can trigger spurious ejections. Rare with `phi_res*m_acc`,
   common with a fixed floor. Fix: sentinel them per branch.

8. **Penarrubia+10 tracks are not calibrated below f_b ~ 1e-3.**
   `ev.Dekel2` is numerically stable to 1e-5 (internal `A` stays < 1),
   but that is extrapolation. Do not run the Dekel branch to
   `phi_res = 1e-5` just because the Green branch can.

9. **`init.Dekel_fromMAH` redraws its scatter every snapshot** — 0.2 dex
   on M* and 0.1 on c_-2/c_-2,DMO — so Dekel parameters along a branch
   carry uncorrelated snapshot noise. Upstream behaviour, kept
   deliberately. Changing it is a separate science decision.

10. **int16 `ParentID` did not actually corrupt existing data.** The
    44869-branch `sdanieli` tree has max `ParentID` = 32767 with only
    113 entries there and zero below -1, i.e. a genuine branch id, not
    saturation — deep branches are discovered last and rarely become
    parents. int32 is correct insurance, not a repair.

## Conventions

- Tree filenames **must** use `%.3f` for the host mass:
  `SubEvo.py` parses them with `tree\d+_lgM(\d+\.\d{3})\.npz`.
- `VirialRadius` in evolved output is the **tidal** radius after
  stripping, not a virial radius (`SubEvo`); `SatEvo` instead stores
  `s.rh` of the stripped Dekel halo. Different meanings — check which
  file you are reading.
- `SubEvo`'s `concentration` output is unchanged from the tree.
- Post-processing lives in
  `ELVES-Dwarf/script/SatGen_script/subhalo.py`
  (`read_subhalos` for `SatEvo`, `read_subhalos_green` for `SubEvo`).

## Smoke test

```python
# cd /home/jiaxuanl/Research/SatGen
import os; os.environ['SLURM_ARRAY_TASK_ID'] = '0'
import TreeGen as T
T.lgM0 = 11.50; T.lgMres = 9.0          # coarse -> ~3 s/tree
T.outfile1 = '/tmp/tree%i_lgM%.3f.npz'
T.loop(0)
```

Checks worth repeating: `ParentID.dtype == int32`; no all-empty rows
(`np.all(mass < 0, axis=1)`); `VirialRadius[0,0] / Rvir(M0, Delta=200)`
should be ~1.2552 at z=0.
