# Handoff — electric motor on a 1/8 sector with antiperiodic symmetry (pde library)

> Focused on **your `pde` Python implementation only** (the NGSolve `alessio/` solver is intentionally out of scope here).
> Every code reference carries `file:line`, read first-hand. The one non-obvious claim — that the sector coupling is **antiperiodic** — is **proven numerically** (§5), correcting an earlier note that mislabeled it "periodic". Repo root: `C:/Users/Z006U5Y9/Documents/GitHub/fem`.

---

## 1. TL;DR

The pde-library magnetostatic motor solver exists in two forms that share almost all assembly code:

- **`SeminarEM/mag_vec_pot.py`** — the **full 360° motor** (48 coils, 16 magnets). Loads `meshes/motor.npz`. Only BC: **penalty Dirichlet** on `stator_outer`. No symmetry. This is the one whose exported HTML you were looking at earlier.
- **`SeminarEM/mag_vec_clean.py`** — the **1/8 = 45° sector** (`pizza = True`). Loads `meshes/motor_pizza.npz`. Imposes **antiperiodic** coupling across the two radial cut edges (`left`↔`right`) and homogeneous Dirichlet on `stator_outer`, both through a single **restriction/prolongation matrix `RS`**. This is the file your question is about.

The symmetry mechanism in the pde path is **fundamentally different** from a penalty or a Lagrange constraint: it builds a tall sparse matrix `RS` that maps a *reduced* set of independent DOFs to the full DOF vector, and solves the **Galerkin-projected** system `RS @ A @ RS.T`. The antiperiodic sign lives in one term: `R_L - R_R`.

Key constants (verified): sector `pizza = True` (`mag_vec_clean.py:22`); `ORDER = 2` → P2 elements (`:37`, `:117-122`); 48 coils (`:207`), 16 magnets (`:214`) — though currents and (effectively) the nonlinearity are disabled in this file (see §7).

---

## 2. Mesh & where the sector comes from

- Loads the **sector** mesh: `motor_npz = np.load('../meshes/motor_pizza.npz', allow_pickle=True)` (`mag_vec_clean.py:24`). This file exists on disk (`PROJECTS/meshes/motor_pizza.npz`).
- But it does **not** mesh from the npz arrays directly. It pulls the OCC geometry out of the npz and **re-meshes with NGSolve/Netgen**, refining twice:
  - `geoOCC = motor_npz['geoOCC'].tolist()` (`:46`), `geoOCCmesh = geoOCC.GenerateMesh()` (`:47`), `ngsolvemesh = ng.Mesh(geoOCCmesh)` (`:48`), `ngsolvemesh.Refine()` ×2 (`:49-50`).
  - Converts to a pde mesh: `MESH = pde.mesh.netgen(ngsolvemesh.ngmesh)` (`:52`).
- `pde.mesh.netgen` (`pde/mesh_class.py:244`) calls `from_netgen` (`:216`) which reads points/edges/triangles, the 1D boundary-region names and 2D material names (`GetBCName`/`GetMaterial`, `:233`/`:237`), **and the periodic point pairs** `identifications = np.array(geoOCCmesh.GetIdentifications())` (`:239`); these are stored on the mesh as `self.identifications` (`mesh_class.py:131`).
- The geometry comes with named boundary edges including `stator_outer`, `left`, `right` (used at `mag_vec_clean.py:171`), and material regions matched by the selectors `linear`/`nonlinear`/`rotor` (`:54-56`).

> The wedge is one pole of a 4-pole-pair machine → 45° = 1/8. The 45° span / pole-pair count live in the **geometry generator** (`PROJECTS/meshes/makeGeometry.py`, the `achtel` branch), not in this solver file; this handoff treats the sector mesh as a given input.

---

## 3. Identifying the cut-edge DOF pairs (`makeIdentifications`)

`makeIdentifications(MESH)` (`mag_vec_clean.py:58`) turns Netgen's raw point identifications into matched **point** and **edge** DOF pairs across the two radial cuts:

- Raw pairs: `a = np.array(MESH.geoOCCmesh.GetIdentifications())` (`:60`).
- For each pair it computes the squared radius of both endpoints (`c0[i] = point0[0]**2+point0[1]**2`, `:69`; same for `c1`, `:70`) and **sorts the pairs by radius** `ind0 = np.argsort(c0)` (`:72`) so the two sides are matched in consistent radial order (inner→outer along the cut).
- It then builds matched **edges** along each side by pairing consecutive sorted points: `edges0` from side-0 points (`:77-78`), `edges1` from side-1 points (`:79-80`), and finds each edge's index in the mesh via `MESH.EdgesToVertices` (`:88-90`).
- Returns `ident_points` (matched vertex-DOF pairs, `:94`) and `ident_edges` (matched edge-DOF pairs, `:96`). For **P2** these are concatenated (vertex DOFs + `MESH.np +` edge DOFs) at `:167`.
- `i0 = ident[:,0]; i1 = ident[:,1]` (`:169`) are the two columns: **`i1` = the `left`-side DOFs, `i0` = the `right`-side DOFs** (this is how they are passed to `assembleR` at `:172-173`).

> ⚠️ **Runtime hazard (verified):** `makeIdentifications` reads `MESH.geoOCCmesh` (`:60`), but `pde.mesh.netgen` has its `cls.geoOCCmesh = geoOCCmesh` assignment **commented out** (`pde/mesh_class.py:245`). So `MESH.geoOCCmesh` does not exist on a mesh built via `pde.mesh.netgen` → this line will raise `AttributeError` as written. Either the attribute must be set manually after `:52`, or `GetIdentifications()` should be read from the local `geoOCCmesh`/`ngsolvemesh` (which *are* in scope). **Treat this sector path as needing a one-line fix before it runs.** (The mesh *does* store the pairs as `MESH.identifications` (`mesh_class.py:131`), so the cleanest fix is to read from there.)

---

## 4. `assembleR` — the restriction-matrix primitive (the core tool)

All of the BC/symmetry machinery is built from one function: `pde.h1.assembleR(MESH, space, edges='', listDOF=empty)` (`pde/h1/assemble.py:208`).

What it does:
- Resolve which boundary edges are meant: a name string → `getIndices(MESH.regions_1d, edges)` (`:217`); empty → all boundary edges (`:215`); a raw DOF array is taken as-is when there are no named regions (`:220`).
- Find the global DOFs living on those edges: `indices = np.in1d(MESH.Boundary_Region, ind_edges)` (`:225`), `LIST_DOF = np.unique(... ['B']['LIST_DOF'][indices,:])` (`:228`); the complement is `LIST_DOF2 = np.setdiff1d(np.arange(sizeM), LIST_DOF)` (`:229`).
- **Override:** if `listDOF` is given, it replaces `LIST_DOF` (`:231-232`) — this is how `R_L`/`R_R` are pinned to the *ordered, matched* pair DOFs `i1`/`i0` from §3 rather than an unordered edge set.
- Build selection matrices from the identity: `D = sp.eye(sizeM)` (`:234`), `R1 = D[:,LIST_DOF]` (`:235`), `R2 = D[:,LIST_DOF2]` (`:236`), and **return `(R1.T, R2.T)`** (`:238`).

So `assembleR` returns **two row-selection operators**: `R1` picks the named DOFs, `R2` picks **everything else** (the complement). A vector restricted by `R1` is "values on those DOFs"; multiplying back by `R1.T` scatters them into the full vector. This is the building block for both the symmetry coupling and the Dirichlet-by-omission below.

---

## 5. How the (anti)symmetry + Dirichlet are applied — `RS`

The sector branch (`if pizza:`, `mag_vec_clean.py:163`) assembles `RS`:

```python
R_out, R_int = assembleR(MESH, poly, edges='stator_outer,left,right')   # :171
R_L,  R_LR  = assembleR(MESH, poly, edges='left',  listDOF=i1)          # :172
R_R,  R_RR  = assembleR(MESH, poly, edges='right', listDOF=i0)          # :173
...
R_L = R_L[ind1,:];  R_R = R_R[ind1,:]   # drop the 2 corner pairs       # :180-181
RS = bmat([[R_int], [R_L - R_R]])                                       # :184
```

Reading each piece (all verified):

1. **`R_int` = the interior basis.** Because `assembleR` returns `(selected, complement)` (§4), the second return value `R_int` is the **complement** of `stator_outer ∪ left ∪ right` — i.e. all DOFs **not** on those boundaries (`:171`). Stacking `R_int` as the first block of `RS` means the full solution is expressed purely from interior DOFs **plus** the coupling rows below — the `stator_outer`, `left`, and `right` DOFs are **not** independent unknowns. Dropping `stator_outer` from the basis **is** the homogeneous Dirichlet `u=0` there (Dirichlet-by-omission), which is why the penalty term is commented out here (§7).

2. **`R_L`, `R_R` = the matched cut-edge selectors.** Each is pinned via `listDOF` to the **ordered** matched DOFs: `R_L` to `i1` (left side), `R_R` to `i0` (right side) (`:172-173`), so row `k` of `R_L` and row `k` of `R_R` are the **same physical point** on the two cuts.

3. **Corner removal.** `corners = np.r_[0, ident_points.shape[0]-1]` (`:177`), `ind1 = np.setdiff1d(..., corners)` (`:178`), then `R_L`/`R_R` are sliced to drop the first/last (the two wedge corners where the cut meets the inner/outer arc) (`:180-181`) — those are handled by the Dirichlet/other coupling and would otherwise be over-constrained.

4. **The coupling block `R_L - R_R`** (`:184`). This is the antiperiodic link. Each such row, as a **prolongation basis vector** (used via `w = RS.T @ wS`, `:285`), sets the left DOF to `+c` and the right DOF to `−c` for a shared reduced unknown `c`.

**Antiperiodic — proven, not asserted.** I reconstructed the exact `RS = bmat([[R_int],[R_L-R_R]])` + `w = RS.T @ wS` in isolation and checked the result:

```
full w     = [10. 20.  7.  9. -7. -9.]   # interior=[10,20], coupling=[7,9]
left  (i1) = [ 7.  9.]
right (i0) = [-7. -9.]
u_left == -u_right (ANTIPERIODIC)?  True
u_left == +u_right (periodic)?      False
```

So the cut coupling is **`u_left = −u_right` (antiperiodic)**. (An earlier note called `R_L−R_R` "plain periodic" — that reasoning treated it as a constraint `R_L−R_R=0` ⟹ equal DOFs; but in this code `RS` is a **prolongation** of independent unknowns, where the `−` produces the sign flip. The pde sector solver is antiperiodic, consistent with the NGSolve solver.)

**How `RS` enters the solve** (Newton, P2, `:272-313`): the tangent and gradient are **projected** into the reduced space each iteration —
```python
gssu = RS @ gss(u) @ RS.T     # :276   reduced Hessian (SPD, sized = #independent unknowns)
gsu  = RS @ gs(u)             # :277   reduced gradient
wS   = chol(gssu).solve_A(-gsu)   # :280   CHOLMOD solve in reduced space
w    = RS.T @ wS              # :285   prolong back to the full DOF vector (applies BC+antiperiodicity)
u    = u + alpha*w            # :309   Armijo line search at :305-307
```
Convergence is measured on the projected gradient `np.linalg.norm(RS @ gs(u))` (`:313`). The projection enforces **both** the Dirichlet (omitted DOFs) and the antiperiodicity (sign-flipped coupling rows) exactly, with no penalty parameter.

---

## 6. The physics / weak form (shared with the full-motor file)

Standard 2D magnetostatic A_z formulation on H1 (P2 here). Assembled operators (`:152-159`): mass `phi_H1`, stiffness blocks `dphix_H1/dphiy_H1`, boundary trace `phi_H1b`, L2 `phi_L2`; quadrature weight diagonals `D_order_*` (`:186-188`).

- Stiffness: `Kxx = dphix_H1 @ D_order_dphidphi @ dphix_H1.T` (`:191`), `Kyy` (`:192`).
- Region masks via `pde.int.evaluate(..., regions=...).diagonal()`: `fem_linear` over `'*air,*magnet,shaft_iron,*coil'` (`:199`, selector `:54`), `fem_nonlinear` over `'stator_iron,rotor_iron'` (`:200`, `:55`).
- Nonlinear iron law imported from `../mixed.EM/nonlinLaws.py` (`:132`); **note `all_linear = 1` there** (`PROJECTS/mixed.EM/nonlinLaws.py:8`, branch `:215`) so `f*_nonlinear` are overridden to the **linear** vacuum-style law — i.e. as committed this sector run is effectively linear.
- Coil current source `aJ` from `j3` per `coil<i+1>` (`:207-209, 224`); magnet remanence `aM` from `m_new` per `magnet<i+1>` (`:214-227`).
- Newton system uses `gss` (tangent, `:253`), `gs` (residual, `:262`), `J` (energy, `:267`). In all three the **penalty term is commented out** (`+ penalty*B_stator_outer`, `:260/:265/:269`) because Dirichlet is enforced structurally by `RS` (§5).

---

## 7. State as committed — what's on / off

- **Sector mode ON:** `pizza = True` (`:22`) → `motor_pizza.npz` + the `RS` antiperiodic branch.
- **Coil currents OFF:** `Ja = 0*Ja`, `J0 = 0*J0` (`:210-211`) → no winding excitation; the only source is the permanent magnets (`aM`). This is a magnet-only / cogging-style run.
- **Effectively linear:** `all_linear = 1` in the imported `nonlinLaws` (`mixed.EM/nonlinLaws.py:8`) collapses the iron law to linear.
- **Penalty Dirichlet OFF (by design):** commented in `gss`/`gs`/`J` (`:260/:265/:269`); Dirichlet is via `RS`'s omission of `stator_outer` DOFs instead.
- **`B_stator_outer` is still assembled** (`:196-197`) but unused in the sector path (leftover from the full-motor path).
- Post-solve it plots `|B|²`-like field `(ux-…)²+(uy+…)²` (`:322`) and the potential `u` (`:326`), then hits a bare `stop` (`:328`) — rotation/torque code below it is commented out (`:333-348`).

---

## 8. Contrast: the full-motor file (`mag_vec_pot.py`)

For reference, the non-sector sibling (`SeminarEM/mag_vec_pot.py`) is the **full 360° motor** and has **no symmetry code at all**:
- Loads `meshes/motor.npz` (`:23`), 48 coils (`:87`), 16 magnets (`:91`).
- BC is a **penalty Dirichlet on `stator_outer`** (the opposite choice from the sector file): `D_stator_outer = pde.int.evaluateB(MESH, edges=ind_stator_outer)` (`:186`), `B_stator_outer = phi_H1b @ D_stator_outer @ phi_H1b.T` (`:187`), `penalty = 1e10` (`:189`), added into `gss`/`gs`/`J` (`:252/:257/:262`).
- Edge/region tagging idiom (reused everywhere): `getIndices` substring match (`pde/tools/getIndices.py:3`); element region tag `MESH.t[:,3]` with `np.isin` (`mag_vec_pot.py:67`); **edge** region tag `e[:,2]` with `np.isin` (`:84`).
- Newton step `w = chol(gss(u)).solve_A(-gsu)` (`:268`) with Armijo (`:310-311`) — same solver pattern, but solving the **full** unconstrained system (no `RS`).

So the two differ in exactly two ways: full-vs-sector mesh, and **penalty Dirichlet (full) vs. structural `RS` Dirichlet+antiperiodicity (sector)**.

---

## 9. File map (concern → file:line)

| Concern | File:line |
|---|---|
| **sector solver** | `PROJECTS/SeminarEM/mag_vec_clean.py` |
| sector flag `pizza=True` | `:22` |
| load sector mesh `motor_pizza.npz` | `:24` |
| OCC re-mesh + 2× Refine | `:46-50` |
| build pde mesh from netgen | `:52` |
| material/region selectors | `:54-56` |
| `makeIdentifications` (cut-pair DOFs) | `:58`, sort-by-radius `:72`, returns `:98` |
| ⚠️ `MESH.geoOCCmesh` (broken) | `:60` (vs `pde/mesh_class.py:245`) |
| `i1`=left DOFs / `i0`=right DOFs | `:169` |
| P2 DOF count (np+NoEdges) | `:122`, ident concat `:167` |
| `R_int` = interior (complement) | `:171` |
| `R_L` (left, listDOF=i1) / `R_R` (right, listDOF=i0) | `:172`, `:173` |
| corner removal | `:177`, `:178`, `:180`, `:181` |
| **`RS = bmat([[R_int],[R_L-R_R]])` (antiperiodic)** | `:184` |
| reduced Hessian / gradient | `:276`, `:277` |
| CHOLMOD reduced solve | `:280` |
| **prolong `w = RS.T@wS`** (applies BC+antisym) | `:285` |
| Newton loop / Armijo / convergence | `:272`, `:305-307`, `:313` |
| penalty commented (Dirichlet via RS) | `:260`, `:265`, `:269` |
| coils & currents zeroed | `:207-209`, `:210-211` |
| magnets `aM` | `:214-227` |
| nonlinLaws import; all_linear=1 | `:132`; `PROJECTS/mixed.EM/nonlinLaws.py:8`, `:215` |
| **assembleR primitive** | `pde/h1/assemble.py:208` |
| edge→DOF (`in1d`, `LIST_DOF`, complement) | `:225`, `:228`, `:229` |
| listDOF override | `:231-232` |
| R1=selected, R2=complement; return (R1.T,R2.T) | `:235`, `:236`, `:238` |
| **mesh container** | `pde/mesh_class.py` |
| `__init__` stores `identifications` | `:31`, `:131` |
| `from_netgen` reads GetIdentifications | `:216`, `:239`, `:241` |
| `netgen` classmethod; geoOCCmesh NOT stored | `:244`, `:245` |
| `EdgesToVertices` (+ boundary marker col 2) | `:71`, `:80-81`, `:112` |
| `np`, `NoEdges` | `:124`, `:115` |
| `Boundary_Region` (edge region tag) | `:118` |
| **full-motor sibling** | `PROJECTS/SeminarEM/mag_vec_pot.py` |
| full motor: motor.npz / 48 / 16 | `:23`, `:87`, `:91` |
| penalty Dirichlet stator_outer | `:186`, `:187`, `:189`, `:252` |
| region/edge tagging (t[:,3]/e[:,2]/isin) | `:67`, `:84` |

---

## 10. Gotchas & open questions

### Verified
1. **The sector coupling is ANTIPERIODIC** (`u_left = −u_right`), via `R_L−R_R` used as a prolongation basis — proven numerically (§5). Do not "simplify" it to `R_L+R_R` (that would be plain periodic) or read the `−` as an equality constraint.
2. **`RS` does Dirichlet AND antiperiodicity together.** `stator_outer`/`left`/`right` DOFs are excluded from the independent set by taking the complement `R_int` (`:171`); the antiperiodic pair rows are added back as `R_L−R_R`. Hence the penalty term is (correctly) commented out (`:260/:265/:269`).
3. **`MESH.geoOCCmesh` at `mag_vec_clean.py:60` will fail** — that attribute is not set by `pde.mesh.netgen` (`mesh_class.py:245` is commented). Fix: read identifications from `MESH.identifications` (already stored, `mesh_class.py:131`) or from the in-scope `geoOCCmesh`. Until fixed, the sector file does not run end-to-end.
4. **Run is magnet-only and effectively linear as committed:** currents zeroed (`:210-211`), `all_linear=1` (`nonlinLaws.py:8`). Re-enable both for a loaded, nonlinear solve.
5. **P2 by default** (`ORDER=2`, `:37`): identifications must include edge DOFs, handled at `:167` (`MESH.np + ident_edges`). If you switch to P1, that concat must be skipped (the `if ORDER==2` guard already does this).

### Open / not established here
- **The 45°/pole-pair geometry** is produced upstream in `PROJECTS/meshes/makeGeometry.py` (the `achtel` branch writes the OCC geometry into the npz); how `motor_pizza.npz` specifically was generated (params, which commit) was not traced in this pass.
- **Whether `motor_pizza.npz`'s stored `geoOCC` regenerates an identical mesh** to what produced its identifications — the re-mesh + 2×Refine (`:49-50`) happens *after* load, so node numbering is freshly generated; `makeIdentifications` relies on `GetIdentifications()` from that regenerated mesh being consistent. Worth a sanity check (count of identified pairs vs. nodes on a cut).
- **Corner DOF handling** beyond dropping the two endpoints (`:180-181`) — whether the inner/outer-arc corner of the wedge is left free or pinned was not separately verified.
