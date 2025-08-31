Here I tried to document how I implemented this method.

### Based on Jiang–Gast–Teran, ACM TOG 2017  

The code simulates a bundle of discrete, inextensible yarns (curves) in two–dimensions.  
Each yarn is a chain of **type-2** Lagrangian mesh vertices.  
Between every two consecutive type-2 vertices we place one **type-3** quadrature particle that stores  

- the deformation gradient $F \in \mathbb{R}^{2\times 2}$, 
- the initial (undeformed) material frame $D = [D₁ | D₂]$, and  
- the current (deformed) material frame $d = [d₁ | d₂] = F D$.

The constitutive response is derived from the **anisotropic hyperelastic potential** for curves (JGT17 §3.3.1) with an additional **Mohr–Coulomb plastic return mapping** for frictional contact (§4.5.2).  


### 2. Symbols  
| Symbol | Dimension | Meaning |
|---|---|---|
| **x₂** | ℝ² | position of type-2 vertex |
| **x₃** | ℝ² | position of type-3 quadrature particle |
| **F** ∈ ℝ²ˣ² | deformation gradient |
| **D** ∈ ℝ²ˣ² | initial material directions |
| **d** ∈ ℝ²ˣ² | current material directions |
| **R** ∈ ℝ²ˣ² | upper-triangular stretch tensor from QR(F) |
| **Q** ∈ ℝ²ˣ² | orthogonal rotation tensor from QR(F) |
| **r₁₁, r₁₂, r₂₂** | scalars | entries of **R** |
| **k, γ, μ, λ** | scalars | material parameters |

---

### 3. Algorithmic Steps  

#### 3.1 `initialize()`
- Places **N_Line** yarns horizontally, spacing **dlx**.  
- For every segment between two type-2 vertices  
- type-3 particle is centered:  
  $$
  \mathbf x_3 = \frac{\mathbf x_2^{(l)} + \mathbf x_2^{(r)}}{2}
  $$  
-  initial frame  
  $$
  \mathbf D = \Bigl[\, \mathbf x_2^{(r)} - \mathbf x_2^{(l)} \;\Big|\; \mathbf R_{90°}(\mathbf x_2^{(r)} - \mathbf x_2^{(l)})\Bigr], \qquad \mathbf R_{90°} = \begin{bmatrix}0&-1\\1&0\end{bmatrix}.
  $$  
- inverse stored once: $\mathbf D^{-1}$.



#### 3.2 `Particle_To_Grid()`
APIC transfer of mass and momentum from particles to grid nodes.  
For each particle $p$ of either type, let  
$$
[\mathbf x_p,\; \mathbf v_p,\; \mathbf C_p,\; m_p]
$$  
be its state.  
For every grid node $i$ in the 3×3 stencil centered at $\lfloor\mathbf x_p/\Delta x-0.5\rfloor$ compute weight  
$$
w_i(\mathbf x_p) = N\!\left(\frac{\mathbf x_p}{\Delta x} - \mathbf x_i\right)
$$  
with quadratic B-spline $N$ (this comes from MLS-MPM).
Accumulate mass and velocity:
$$
m_i \mathrel{+}= w_i\, m_p, \qquad
\mathbf v_i \mathrel{+}= w_i\, m_p \left(\mathbf v_p + \mathbf C_p(\mathbf x_i - \mathbf x_p)\right).
$$  
Finally set $\mathbf v_i \leftarrow \mathbf v_i/m_i$.

#### 3.3 `Grid_Force()`
Computes **nodal forces** from the stored elastic energy $\psi$.

For every type-3 particle $p$:

1. **QR decomposition** (in 2D)  
   $$
   \mathbf F = \mathbf Q\mathbf R,\qquad
   \mathbf R = \begin{bmatrix}r_{11}&r_{12}\\0&r_{22}\end{bmatrix}.
   $$

2. **Elastic potential derivatives**  
   Energy split  in 2D becomes:
$$
   \psi(\mathbf R) = \underbrace{\tfrac12 k(r_{11}-1)^2}_{f(R_1)} +
                     \underbrace{\tfrac12 \gamma r_{12}^2}_{g(R_2)} +
                     \underbrace{\tfrac12(2\mu+\lambda)\left(\log r_{22}\right)^2}_{h(R_3)} \quad(r_{22}\le 1).
   $$
   Hence the derivatives are simply:$$
   \frac{\partial\psi}{\partial r_{11}} = k(r_{11}-1),\qquad
   \frac{\partial\psi}{\partial r_{12}} = \gamma r_{12},\qquad
   \frac{\partial\psi}{\partial r_{22}} = (2\mu+\lambda)\frac{\log r_{22}}{r_{22}}.
   $$
   Collect into symmetric matrix  
$$
   \mathbf A = \begin{bmatrix}
   A_{00} & A_{01}\\[2pt]
   A_{10} & A_{11}
   \end{bmatrix}=\begin{bmatrix}
   k(r_{11}-1) & \gamma r_{12}\\[2pt]
   \gamma r_{12} & (2\mu+\lambda)\frac{\log r_{22}}{r_{22}}
   \end{bmatrix}.
   $$
   

3. **First Piola–Kirchhoff-like stress**  can then be computed as: 
   $$
   \mathbf P = \mathbf Q\,\mathbf A\,\mathbf R^{-\top}.
   $$

4. **Force assembly**  
   For each grid node $i$ in the stencil of either attached type-2 vertex or the quadrature particle, add force:
   $$
   \mathbf f_i \mathrel{+}= w_i\,V_p\,\mathbf P\,\mathbf D^{-\top}\mathbf e_1
   $$  
   (left vertex) and :
   $$
   \mathbf f_i \mathrel{-}= w_i\,V_p\,\mathbf P\,\mathbf D^{-\top}\mathbf e_1
   $$  
   (right vertex) plus contributions from the normal component.

Additionally, a **bending spring** between every second type-2 vertex is applied for stability:  
$$
\mathbf f_{\text{bend}} = -k_{\text{bend}}\left(|\mathbf x_j-\mathbf x_i|-2\ell_0\right)\frac{\mathbf x_j-\mathbf x_i}{|\mathbf x_j-\mathbf x_i|},
$$  
distributed to the grid using the same APIC weights.

#### 3.4 `Grid_Collision()`
- Adds gravity $\mathbf g = (0,-9.8)$.  
- Applies **attractor** force  
  $$
  \mathbf f_{\text{attr}} = \frac{\mathbf x_{\text{mouse}}-\mathbf x_i}{\|\mathbf x_{\text{mouse}}-\mathbf x_i\|+0.01}\,s_{\text{attr}}.
  $$  
- Implements **three circular obstacles** using simple reflection and friction.  
- Enforces zero-velocity boundary conditions on domain edges.

#### 3.5 `Grid_To_Particle()`
Transfers grid velocity and affine matrix back to particles via APIC inverse stencil.

#### 3.6 `Update_Particle_State()`
- Type-3 velocity and position are re-interpolated from adjacent type-2 vertices:  
  $$
  \mathbf v_3 = \tfrac12(\mathbf v_2^{(l)}+\mathbf v_2^{(r)}),\qquad
  \mathbf x_3 = \tfrac12(\mathbf x_2^{(l)}+\mathbf x_2^{(r)}).
  $$  
- Updates **d₁** (edge vector) and **d₂** (normal vector)  
  $$
  \mathbf d_1 = \mathbf x_2^{(r)}-\mathbf x_2^{(l)},\qquad
  \mathbf d_2 \leftarrow \mathbf d_2 + \Delta t\,\mathbf C_3\,\mathbf d_2.
  $$  
- Re-computes deformation gradient  
  $$
  \mathbf F = \mathbf d\,\mathbf D^{-1}.
  $$

#### 3.7 `Return_Mapping()`
Implements **Mohr–Coulomb plasticity** for 2-D curves:

1. If $r_{22}>1$ → free expansion, set $r_{12}=0,\;r_{22}=1$.  
2. Else compute yield strength  
   $$
   \tau_{\text{yield}} = c_f(1-r_{22}).
   $$  
3. Clamp shear  
   $$
   r_{12} \leftarrow \text{clamp}(r_{12}, -\tau_{\text{yield}}, \tau_{\text{yield}}).
   $$  
4. Reconstruct  
   $$
   \mathbf R_{\text{new}} = \begin{bmatrix}r_{11}&r_{12}\\0&r_{22}\end{bmatrix},\qquad
   \mathbf F \leftarrow \mathbf Q\mathbf R_{\text{new}}.
   $$

### 4. Material Parameters  
| Symbol | Value | Meaning |
|---|---|---|
| $k$ | 1000 | fiber stretch stiffness |
| $\gamma$ | 500 | fiber shear stiffness |
| $(\mu,\lambda)$ | $(0.4E,0.6E)$ | Lamé coefficients for cross-section |
| $c_f$ | 0.05 | friction coefficient |
