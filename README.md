# FARMS MuJoCo

This package provides the MuJoCo simulation backend for the FARMS
amphibious simulation framework. It handles model construction from SDF
to MJCF, physics simulation, sensor data collection, swimming
hydrodynamics, and interactive visualization.

## Key Features

- **SDF to MJCF conversion** — converts animat SDF models to MuJoCo
  MJCF format with automatic sensor, actuator, and collision setup
- **Multi-animat simulation** — supports simultaneous simulation of
  multiple animats (e.g. schooling) with per-animat collision filtering
  that prevents self-collisions while allowing inter-animat collisions
- **Extension system** — modular task extensions for controllers,
  swimming forces, cameras, and visualization, loaded from animat
  configuration
- **Swimming hydrodynamics** — quadratic drag and buoyancy forces with
  implicit (backward Euler) formulation for numerical stability (see
  `swimming/README.md`)
- **Sensor data collection** — link positions, orientations, velocities,
  joint states, contacts, and external forces copied from MuJoCo to
  FARMS sensor arrays each timestep
- **Interactive viewer** — MuJoCo passive viewer with keyboard controls
  (pause, speed, step), or dm_control application viewer
- **Camera support** — video recording and camera following
- **Mesh loading cache** — meshes are cached at the module level so
  that multiple animats sharing the same model load each mesh only once
- **Headless texture skipping** — in headless mode, mesh textures are
  not loaded, significantly reducing setup time

## Architecture

### Simulation Pipeline

```
farmsim (entry point)
  └── simulation_setup → MuJoCoSimulation.from_experiment
        ├── setup_mjcf_xml (SDF → MJCF)
        │     ├── sdf2mjcf (arena, water, animats)
        │     │     └── mjc_add_link (per-link: meshes, joints, sensors)
        │     └── Compiler options (integrator, solver, etc.)
        ├── ExperimentTask (task loop)
        │     ├── extract_extensions (controllers, swimming, cameras)
        │     ├── initialize_episode (maps, sensors, control)
        │     └── before_step / after_step (per-timestep callbacks)
        └── run / iterator (simulation loop)
```

### Module Overview

| Module | Description |
|--------|-------------|
| `simulation/simulation.py` | `Simulation` class — creates physics, runs the simulation loop, handles viewer |
| `simulation/task.py` | `ExperimentTask` — per-timestep callbacks: sensor update, extension stepping, joint control |
| `simulation/mjcf.py` | SDF to MJCF conversion: mesh loading (with cache), collision filtering, actuator/sensor setup |
| `simulation/physics.py` | Data transfer: copies MuJoCo state (positions, velocities, contacts) to FARMS sensor arrays |
| `simulation/extensions.py` | Viewer extensions: camera following, trail rendering, center-of-mass tracking |
| `swimming/drag.pyx` | Cython: quadratic drag, buoyancy, water properties, implicit drag solver |
| `swimming/extension.py` | `SwimmingExtension` — computes and applies fluid forces each timestep |
| `sensors/sensors.pyx` | Cython: contact and muscle sensor data extraction |

### Per-Timestep Loop

Each simulation step follows this sequence:

1. **`before_step`** (task):
   - Update sensors (copy MuJoCo state to FARMS arrays)
   - Step all extensions (controllers, swimming):
     - Controller: step drive (boids/PID) → step network (CPG ODE) → compute joint torques
     - Swimming: compute drag/buoyancy → rotate to global frame → apply to `xfrc_applied`
   - Apply joint control (position, velocity, torque) to MuJoCo actuators
   - Apply spring stiffness, damping, spring references

2. **`mj_step`** (MuJoCo):
   - Integrate equations of motion (implicitfast)
   - Resolve contacts and constraints

3. **`after_step`** (task):
   - Update viewer (if not headless)

### Substepping

The simulation supports substepping: `cb_sub_steps` physics steps per
control callback. The control timestep is `simulation_options.physics.timestep`,
and the physics timestep is `timestep / (cb_sub_steps * num_sub_steps)`.
Extensions can opt into substepping via the `substep` flag.

## Mesh Loading

When multiple animats share the same mesh files (e.g. schooling),
each unique mesh is loaded only once and reused for subsequent animats.
The cache is stored at the module level in
`farms_mujoco/simulation/mjcf.py` (`_TRIMESH_CACHE`, `_WAVEFRONT_CACHE`).
The `clear_mesh_cache()` function can be called to invalidate the cache
between simulation runs.

In headless mode (`simulation_options.runtime.headless = True`):
- Mesh geometry is loaded without textures (using
  `trimesh.exchange.obj.load_obj()` instead of `trimesh.load_mesh()`)
- Wavefront texture parsing is skipped
- MuJoCo's `discardvisual` compiler flag is set to `True`

## Collision Filtering

Each animat is assigned a unique collision type bit (`contype`) using
`2**((animat_i % 30) + 1)`, with `conaffinity` set to all bits except
its own. This prevents self-collisions while allowing inter-animat
collisions. The first 30 animats get unique bits; beyond that, animats
cycle through bits, and those sharing a bit will not collide with each
other but will still collide with all others.

The arena floor uses `contype=1, conaffinity=2**31-1` (all bits set)
to collide with everything.

## Swimming Hydrodynamics

See `swimming/README.md` for the fluid model documentation and
`swimming/NOTES.md` for developer notes on the implicit drag formulation.
