"""
Probability-space (p-space) averaging, symmetrization and bootstrap error for the
ketoprofen / POPC WTM-eABF PMFs.

Method follows Kang et al., "Convergence is not correctness" (Nat. Commun. 2026),
Eqs 14-18: each replica's PMF is converted to a normalized probability, replicas
are averaged in probability space, converted back, and the error is the bootstrap
standard deviation across the resampled ensemble.  No profile is anchored at a
reference state -- the normalization fixes the additive constant, so uncertainty
is spread across the whole CV range rather than pinned to zero at an anchor.

Two additions for the membrane system:
  * symmetrization about z=0 (the bilayer is symmetric; exploiting it halves the
    statistical error), done in p-space for consistency with the consensus.
  * an error decomposition:
      sigma_boot   -- inter-seed bootstrap on the symmetrized profiles
      sigma_asym   -- SEM of the residual (antisymmetric) asymmetry across seeds
      sigma_quad   = sqrt(sigma_boot^2 + sigma_asym^2)         (independent-add)
      sigma_nested -- a single two-stage bootstrap that resamples seeds AND the
                      left/right reflection, folding both variance components in
                      without assuming independence.
    quad and nested are reported side by side.

The grid is symmetric about z=0 (bin centres -40.0 .. +40.0), so reflection is
just array reversal: g(-z) = g[::-1].
"""

import numpy as np
from keto_pmf import KT, BETA


# ---------------------------------------------------------------------------
# probability <-> free energy
# ---------------------------------------------------------------------------
def to_prob(g):
    """Free energy (kcal/mol) -> normalized probability (Eq 14).  Works rowwise
    on the last axis for 1-D or 2-D input."""
    g = np.asarray(g, float)
    gmin = g.min(axis=-1, keepdims=True)
    p = np.exp(-BETA * (g - gmin))
    return p / p.sum(axis=-1, keepdims=True)


def to_free(p):
    """Normalized probability -> free energy (kcal/mol), gauge fixed by the
    normalization (no arbitrary shift).  Rowwise on the last axis."""
    return -KT * np.log(np.asarray(p, float))


def anchor_bulk(g, z, bulk_min=35.0):
    """Shift a free-energy profile so the bulk-water region (|z|>=bulk_min) = 0.
    Cosmetic only (display); never applied before computing sigma."""
    m = np.abs(z) >= bulk_min
    return g - g[..., m].mean(axis=-1, keepdims=True)


def reflect(a):
    """g(-z) on a grid symmetric about z=0 -> array reversal."""
    return a[..., ::-1]


# ---------------------------------------------------------------------------
# per-cell consensus + error
# ---------------------------------------------------------------------------
class CellResult:
    """Holds the symmetrized p-space consensus PMF and its error decomposition
    for one parameter cell (one set of parameters, N seeds)."""
    pass


def _reduce(P, axis, estimator):
    """Combine a stack of probability profiles across `axis` in p-space."""
    if estimator == "median":
        return np.median(P, axis=axis)
    return np.mean(P, axis=axis)


def analyze_cell(g_list, z, B=5000, seed=1234, bulk_min=35.0, estimator="median"):
    """
    g_list : (N, nbins) raw per-seed free-energy profiles (kcal/mol)
    z      : (nbins,) CV grid, symmetric about 0
    estimator : 'median' (default, robust to seed-to-seed well-position jitter --
                matches the "Median (P-space consensus)" of the manuscript) or
                'mean' (Kang et al. literal, but the arithmetic mean of
                probabilities smears misaligned wells and under-deepens them here).
    Returns a CellResult with (all free energies anchored to bulk=0 for display):
      z, N
      pmf            consensus symmetrized p-space PMF
      sigma_boot     inter-seed bootstrap SD
      sigma_asym     asymmetry SEM
      sigma_quad     sqrt(boot^2 + asym^2)
      sigma_nested   two-stage (seed + reflection) bootstrap SD
      boot_ensemble  (B, nbins) inter-seed bootstrap free-energy replicas (bulk-anchored)
      pmf_raw_mean   consensus of the *un*symmetrized profiles (for asym display)
    """
    rng = np.random.default_rng(seed)
    G = np.asarray(g_list, float)
    N, nb = G.shape

    P = to_prob(G)                      # (N, nb) normalized per seed
    Prefl = reflect(P)
    Psym = 0.5 * (P + Prefl)            # symmetrized in p-space (still normalized)

    # --- consensus (Eqs 15-17, robust reducer) ---
    Pbar = _reduce(Psym, 0, estimator)
    pmf = to_free(Pbar)

    # --- inter-seed bootstrap sigma_boot (Eq 18) ---
    idx = rng.integers(0, N, size=(B, N))
    Pb = _reduce(Psym[idx], 1, estimator)       # (B, nb)
    A_boot = to_free(Pb)                        # gauge fixed by normalization
    sigma_boot = A_boot.std(axis=0, ddof=1)

    # --- asymmetry SEM sigma_asym ---
    # antisymmetric residual of each raw seed: a_i(z) = 1/2 (g_i(z) - g_i(-z))
    a = 0.5 * (G - reflect(G))                  # (N, nb), antisymmetric
    sigma_asym = a.std(axis=0, ddof=1) / np.sqrt(N)

    # --- quadrature total ---
    sigma_quad = np.sqrt(sigma_boot**2 + sigma_asym**2)

    # --- unified two-stage bootstrap sigma_nested ---
    # outer: resample seeds; inner: for each drawn seed average two draws from
    # {P_s, reflect(P_s)} -> weight r in {0,0.5,1}.  The replica consensus is left
    # (generally) asymmetric on purpose: left/right disagreement -- including an
    # asymmetry shared across seeds -- inflates the variance.  The true error is
    # symmetric, so we symmetrize the *variance* at the end.
    sidx = rng.integers(0, N, size=(B, N))
    r = rng.integers(0, 2, size=(B, N, 2)).mean(axis=2)          # {0,0.5,1}
    Psel = (1.0 - r)[..., None] * P[sidx] + r[..., None] * Prefl[sidx]
    Pn = _reduce(Psel, 1, estimator)                            # (B, nb), normalized
    A_nest = to_free(Pn)
    var_nest = A_nest.var(axis=0, ddof=1)
    sigma_nested = np.sqrt(0.5 * (var_nest + reflect(var_nest)))

    # unsymmetrized consensus (to visualise residual asymmetry that sym. removes)
    pmf_raw_mean = to_free(_reduce(P, 0, estimator))

    # --- anchor everything to bulk = 0 for display ---
    shift = pmf[np.abs(z) >= bulk_min].mean()
    res = CellResult()
    res.z = z
    res.N = N
    res.pmf = pmf - shift
    res.pmf_raw_mean = pmf_raw_mean - pmf_raw_mean[np.abs(z) >= bulk_min].mean()
    res.sigma_boot = sigma_boot
    res.sigma_asym = sigma_asym
    res.sigma_quad = sigma_quad
    res.sigma_nested = sigma_nested
    res.boot_ensemble = A_boot - A_boot[:, np.abs(z) >= bulk_min].mean(axis=1, keepdims=True)
    return res


# ---------------------------------------------------------------------------
# RMSD (vertically aligned; Kang Eq 19-20)
# ---------------------------------------------------------------------------
def rmsd(a, b, mask=None):
    """RMSD between two PMFs after optimal vertical alignment (subtract mean
    difference).  Optional boolean mask restricts the comparison region."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = a - b
    if mask is not None:
        d = d[mask]
    d = d - d.mean()
    return float(np.sqrt(np.mean(d**2)))


def symmetrize_free(g):
    """Symmetrize a free-energy profile directly in G-space (for reference /
    per-seed profiles used only in RMSD comparisons)."""
    g = np.asarray(g, float)
    return 0.5 * (g + reflect(g))


# ---------------------------------------------------------------------------
# physical observables with bootstrap error
# ---------------------------------------------------------------------------
def observables(res, well=(8.0, 16.0), head=(18.0, 28.0)):
    """Extract barrier / well-depth observables (kcal/mol) from a CellResult,
    with error from the inter-seed bootstrap ensemble.  All relative to bulk=0.

    Returns dict of (value, sigma):
      well_depth        depth of the acyl-region minima below bulk  (positive)
      central_barrier   G(z=0) relative to the wells                (positive)
      center_vs_bulk    G(z=0) relative to bulk
      head_peak         interfacial head-group peak above bulk
    """
    z = res.z
    scal = lambda prof: pmf_scalars(prof, z, well, head)
    mean = scal(res.pmf)
    # propagate via bootstrap ensemble
    ens = {k: [] for k in mean}
    for prof in res.boot_ensemble:
        s = scal(prof)
        for k in s:
            ens[k].append(s[k])
    out = {}
    for k in mean:
        out[k] = (mean[k], float(np.std(ens[k], ddof=1)))
    return out


def pmf_scalars(prof, z, well=(8.0, 16.0), head=(18.0, 28.0)):
    """Barrier / well observables from one PMF profile (bulk=0)."""
    zc = np.argmin(np.abs(z))
    wmask = (np.abs(z) >= well[0]) & (np.abs(z) <= well[1])
    hmask = (np.abs(z) >= head[0]) & (np.abs(z) <= head[1])
    gwell = prof[wmask].min()
    center = prof[zc]
    return dict(well_depth=-gwell, central_barrier=center - gwell,
                center_vs_bulk=center, head_peak=prof[hmask].max())


# ---------------------------------------------------------------------------
# Population-SD error model (consistent with the other systems' "1 SD over
# replicas"), extended to fold in the ketoprofen-specific asymmetry.
# ---------------------------------------------------------------------------
def population_errors(g_list, z, bulk_min=35.0):
    """Population standard deviations (spread of individual replicas), matching
    the other systems' error convention, with the symmetry folded in.

    Returns dict of z-arrays:
      sd_repl   inter-replica SD of the symmetrized, bulk-anchored profiles
                (the direct analogue of the other systems' "1 SD over replicas")
      sd_asym   RMS per-replica asymmetry, 1/2|G(z)-G(-z)| (ketoprofen-specific)
      sd_quad   sqrt(sd_repl^2 + sd_asym^2)
      sd_total  SD across the 2N mirror-image half-profiles {G_i(z)} u {G_i(-z)}
                -- exact total spread; equals sd_quad up to the ddof convention
    """
    G = np.asarray(g_list, float)
    N = G.shape[0]
    Ga = G - G[:, np.abs(z) >= bulk_min].mean(axis=1, keepdims=True)  # anchor each
    Gr = reflect(Ga)
    S = 0.5 * (Ga + Gr)                 # symmetrized per seed
    a = 0.5 * (Ga - Gr)                 # asymmetry per seed (antisymmetric)
    sd_repl = S.std(axis=0, ddof=1)
    sd_asym = np.sqrt((a ** 2).mean(axis=0))          # spread about the true value 0
    sd_quad = np.sqrt(sd_repl ** 2 + sd_asym ** 2)
    pool = np.concatenate([Ga, Gr], axis=0)           # 2N mirror-image profiles
    sd_total = pool.std(axis=0, ddof=1)
    sd_total = np.sqrt(0.5 * (sd_total ** 2 + reflect(sd_total) ** 2))  # enforce symmetry
    return dict(sd_repl=sd_repl, sd_asym=sd_asym, sd_quad=sd_quad, sd_total=sd_total)


def population_observables(g_list, z, bulk_min=35.0, well=(8.0, 16.0), head=(18.0, 28.0)):
    """Observable value (from the per-seed mean) with error = 1 SD over the N
    symmetrized replicas (population flavour, matching the other systems)."""
    G = np.asarray(g_list, float)
    Gs = symmetrize_free(G)
    Gs = Gs - Gs[:, np.abs(z) >= bulk_min].mean(axis=1, keepdims=True)
    per = [pmf_scalars(gs, z, well, head) for gs in Gs]
    keys = per[0].keys()
    out = {}
    for k in keys:
        vals = np.array([p[k] for p in per])
        out[k] = (float(vals.mean()), float(vals.std(ddof=1)))
    return out
