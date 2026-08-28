# Swimming Hydrodynamics

This module implements fluid forces for swimming animats in the FARMS
simulation framework. It computes **quadratic drag** and **buoyancy** forces
on each link of an articulated body and applies them as external forces
to the MuJoCo physics engine.

The drag model uses an **implicit (backward Euler) formulation** for the
velocity-dependent drag forces, which provides unconditional stability for
large timesteps. This allows simulation timesteps to be increased without
the numerical instability that would otherwise arise from the explicit
treatment of quadratic drag.

## Physics Model

### Quadratic Drag

Each link experiences a drag force proportional to the square of its
relative velocity through the fluid:

```
F_drag = -c * v_rel * |v_rel|
```

where:
- `c = viscosity * drag_coefficient` — the drag coefficient (per axis)
- `v_rel = v_link - v_fluid` — the relative velocity of the link with
  respect to the surrounding fluid, expressed in the link's URDF frame

The drag is computed **per axis** in the link's local (URDF) frame, where
the three axes are decoupled:

```
F_i = -c_i * v_i * |v_i|
```

### Torque Drag

Angular drag is similarly quadratic in angular velocity:

```
τ_i = -c_i * ω_i * |ω_i|
```

The torque is computed in the link's CoM frame using angular velocity and
torque drag coefficients.

### Buoyancy

Buoyancy is a position-dependent force that acts upward when a link is
submerged:

```
F_buoyancy_z = -ρ_water * m * g / ρ_link * submersion_fraction
```

where:
- `ρ_water` — fluid density
- `ρ_link` — link density
- `m` — link mass
- `g` — gravitational acceleration
- `submersion_fraction = clamp((surface + height - z) / (2*height), 0, 1)`

### Water Properties

The fluid environment is described by a `WaterProperties` object:

- **Surface height** — the water level (links above this are out of water)
- **Density** — fluid density (for buoyancy)
- **Velocity** — fluid velocity field (for relative velocity computation)
- **Viscosity** — fluid viscosity (scales drag forces)

Two implementations are provided:
- `WaterPropertiesConstant` — uniform, time-invariant properties (default)
- `WaterPropertiesExtension` — spatially-varying properties via callbacks
  (e.g. flow maps from PNG images)

## Choosing Drag Coefficients

The drag coefficient per axis is `c_i = viscosity * |coefficient_i|`. For
physical realism, it should match the hydrodynamic drag on the link:

```
c_i = 0.5 * ρ_water * C_d * A_i
```

where `A_i` is the projected area perpendicular to axis `i`, and `C_d` is
the drag coefficient for the body shape in that direction:

| Direction | `C_d` | Description |
|-----------|-------|-------------|
| Along body (x) | 0.1–0.5 | Streamlined, low drag |
| Lateral (y) | 1.0–1.2 | Bluff, high drag |
| Vertical (z) | 1.0–1.2 | Bluff, high drag |

For a cylindrical link with radius `r` and length `L`:
- `A_x = π * r²` (frontal area)
- `A_y = 2 * r * L` (side area)
- `A_z = 2 * r * L` (top/bottom area)

The anisotropy ratio `c_y / c_x = A_y / A_x = 2*L / (π*r)` depends on the
link's aspect ratio. For a fish-like body (elongated), the lateral drag
is much larger than the longitudinal drag, which is essential for
anguilliform swimming.

## Configuration

Drag is configured per-link in the animat's SDF/options:

```yaml
morphology:
  links:
  - name: link_0
    fluid_interaction: true          # Enable drag/buoyancy for this link
    density: 1000.0                   # Link density (for buoyancy)
    drag_coefficients:
    - - -0.001                        # Linear drag x (force)
      - -0.1                          # Linear drag y
      - -0.1                          # Linear drag z
    - - 0                             # Angular drag x (torque)
      - 0
      - 0
```

Water properties are configured in the arena options:

```yaml
water:
  drag: true                          # Enable drag forces
  buoyancy: true                      # Enable buoyancy
  height: 0                           # Water surface height
  density: 1000.0                     # Fluid density [kg/m³]
  viscosity: 1.0                      # Fluid viscosity
  velocity: [0, 0, 0]                 # Constant fluid velocity
```

## File Structure

| File | Description |
|------|-------------|
| `drag.pyx` | Cython implementation: drag forces, buoyancy, water properties, swimming handler |
| `extension.py` | Python integration: `SwimmingExtension` (MuJoCo callback), water velocity maps |
| `README.md` | User documentation (this file) |
| `NOTES.md` | Developer notes |
