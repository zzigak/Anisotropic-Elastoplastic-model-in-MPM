import taichi as ti

ti.init(arch=ti.gpu)  # Try to run on GPU

quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality
dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality
p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho
E, nu = 8e3, 0.25  # Higher Young's modulus for stiffer, less stretchy yarn
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
material = ti.field(dtype=int, shape=n_particles)  # material id
Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass
gravity = ti.Vector.field(2, dtype=float, shape=())
attractor_strength = ti.field(dtype=float, shape=())
attractor_pos = ti.Vector.field(2, dtype=float, shape=())


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # deformation gradient update
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
        # Hardening coefficient: snow gets harder when compressed
        h = ti.max(0.1, ti.min(5, ti.exp(10 * (1.0 - Jp[p]))))
        if material[p] == 1:  # yarn-like material, stiffer to prevent excessive stretching
            h = 0.5
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(2)):
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0:
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity
            F[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 2) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            # Loop over 3x3 grid node neighborhood
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            # Momentum to velocity
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]
            grid_v[i, j] += dt * gravity[None] * 30  # gravity
            dist = attractor_pos[None] - dx * ti.Vector([i, j])
            grid_v[i, j] += dist / (0.01 + dist.norm()) * attractor_strength[None] * dt * 100
            if i < 3 and grid_v[i, j][0] < 0:
                grid_v[i, j][0] = 0  # Boundary conditions
            if i > n_grid - 3 and grid_v[i, j][0] > 0:
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0:
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0
            
            # COMPLETELY freeze the top boundary - make it like a solid ceiling
            # Any grid point at or near the top is completely immobile
            if j >= n_grid - 5:  # Top 5 grid cells are completely frozen
                grid_v[i, j][0] = 0  # No horizontal movement
                grid_v[i, j][1] = 0  # No vertical movement
    for p in x:  # grid to particle (G2P)
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            # loop over 3x3 grid node neighborhood
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]  # advection


@ti.kernel
def reset():
    n_yarns = 5  # Number of vertical hanging yarns
    particles_per_yarn = n_particles // n_yarns
    yarn_length = 0.6  # Total length of each yarn
    yarn_thickness = 0.001  # Ultra-thin yarn-like strands
    horizontal_spacing = 0.08  # Space between yarns horizontally - closer together
    
    for yarn_id in range(n_yarns):
        for local_id in range(particles_per_yarn):
            i = yarn_id * particles_per_yarn + local_id
            
            # Distribute particles along the yarn length
            # Start from very top of the domain (y ≈ 1.0) going downward
            # First few particles will be in the frozen zone
            y_pos = 0.98 - (local_id / particles_per_yarn) * yarn_length
            
            # Horizontal position with spacing
            x_pos = 0.2 + yarn_id * horizontal_spacing
            
            # Add tiny random perturbation for thickness
            x[i] = [
                x_pos + (ti.random() - 0.5) * yarn_thickness,
                y_pos + (ti.random() - 0.5) * yarn_thickness
            ]
            material[i] = 1  # All particles are yarn material
            v[i] = [0, 0]
            F[i] = ti.Matrix([[1, 0], [0, 1]])
            Jp[i] = 1
            C[i] = ti.Matrix.zero(float, 2, 2)


print("[Hint] Use WSAD/arrow keys to control gravity. Use left/right mouse buttons to attract/repel. Press R to reset.")
gui = ti.GUI("Hanging Yarns - Taichi MLS-MPM", res=512, background_color=0xFFFFFF)
reset()
gravity[None] = [0, -1]

for frame in range(20000):
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == "r":
            reset()
        elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            break
    if gui.event is not None:
        gravity[None] = [0, 0]  # if had any event
    if gui.is_pressed(ti.GUI.LEFT, "a"):
        gravity[None][0] = -1
    if gui.is_pressed(ti.GUI.RIGHT, "d"):
        gravity[None][0] = 1
    if gui.is_pressed(ti.GUI.UP, "w"):
        gravity[None][1] = 1
    if gui.is_pressed(ti.GUI.DOWN, "s"):
        gravity[None][1] = -1
    mouse = gui.get_cursor_pos()
    gui.circle((mouse[0], mouse[1]), color=0x336699, radius=15)
    attractor_pos[None] = [mouse[0], mouse[1]]
    attractor_strength[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        attractor_strength[None] = 1
    if gui.is_pressed(ti.GUI.RMB):
        attractor_strength[None] = -1
    for s in range(int(2e-3 // dt)):
        substep()
    gui.circles(
        x.to_numpy(),
        radius=1.5,
        color=0xED553B,  # Jelly color (orange-red)
    )

    # Change to gui.show(f'{frame:06d}.png') to write images to disk
    gui.show()