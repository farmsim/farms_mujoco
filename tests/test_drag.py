"""Regression tests for the swimming drag model.

These exercise ``drag_forces`` directly with synthetic link/xfrc arrays.
They require the Cython extension to be rebuilt (``pip install -e .``).
"""

import numpy as np
import pytest

from farms_core.sensors.data_cy import LinkSensorArrayCy, XfrcArrayCy
from farms_core.sensors.sensor_convention import sc
from farms_mujoco.swimming.drag import drag_forces, WaterPropertiesConstant


def _run(
        *,
        linear_coefficient=-1.0,
        angular_coefficient=0.0,
        z=0.0,
        height=0.1,
        quat=(0.0, 0.0, 0.0, 1.0),
        linear_velocity=1.0,
        angular_velocity=0.0,
        dt=0.01,
        inertia=None,
        use_buoyancy=False,
        buoyancy_center=(0.0, 0.0, 0.0),
        water=None,
        initial_force=None,
):
    """Run one link through drag_forces and return output + scratch buffers."""
    links = LinkSensorArrayCy(np.zeros((1, 1, sc.link_size)))
    xfrc = XfrcArrayCy(np.zeros((1, 1, 6)))
    if initial_force is not None:
        np.asarray(xfrc.array)[0, 0, :] = initial_force
    link_array = np.asarray(links.array)
    link_array[0, 0, sc.link_com_position_z] = z
    link_array[0, 0, sc.link_urdf_orientation_x:
                    sc.link_urdf_orientation_w+1] = quat
    link_array[0, 0, sc.link_com_orientation_x:
                    sc.link_com_orientation_w+1] = quat
    link_array[0, 0, sc.link_com_velocity_lin_x] = linear_velocity
    link_array[0, 0, sc.link_com_velocity_ang_x] = angular_velocity

    coefficients = np.array([
        [linear_coefficient]*3,
        [angular_coefficient]*3,
    ])
    if water is None:
        water = WaterPropertiesConstant(
            surface=0.0,
            density=1000.0,
            velocity=np.array([0.0, 0.0, 0.0]),
            viscosity=1.0,
        )
    z3 = np.zeros((7, 3))
    z4 = np.zeros((7, 4))
    drag_forces(
        0.0, 0, links, 0, xfrc, 0, coefficients, z3, z4, water,
        1.0, height, 1000.0, -9.81, use_buoyancy,
        dt=dt,
        buoyancy_center=np.array(buoyancy_center, dtype=float),
        inertia=inertia,
    )
    return np.asarray(xfrc.array)[0, 0].copy(), z4[0].copy()


def test_link_orientation_propagation():
    """D5: the caller's urdf2global buffer receives the link quaternion."""
    quat = np.array([0.70710678, 0.0, 0.0, 0.70710678])
    _, urdf2global = _run(quat=quat)
    assert np.allclose(urdf2global, quat)


def test_implicit_linear_drag_is_active():
    """D3/D1: full submersion takes the implicit path, not the explicit one."""
    force, _ = _run(z=-0.2)
    # m=1, dt=0.01, c=1, v=1 -> -0.980486..., not the explicit -1.
    assert force[0] == pytest.approx(-0.9804864072151531, rel=1e-10)


def test_surface_scaling_and_cutoff():
    """D1: drag scales with the submerged fraction and is zero above water."""
    half_force, _ = _run(z=0.0)       # fraction 0.5
    full_force, _ = _run(z=-0.2)      # fraction 1.0
    dry_force, _ = _run(z=0.1)        # fraction 0.0
    assert half_force[0] == pytest.approx(-0.49506163792187863, rel=1e-10)
    assert full_force[0] == pytest.approx(-0.9804864072151531, rel=1e-10)
    assert np.allclose(dry_force, 0.0)


def test_implicit_angular_drag():
    """D2: angular drag uses the inertia-based implicit solve."""
    force, _ = _run(
        linear_coefficient=0.0,
        angular_coefficient=-0.01,
        angular_velocity=100.0,
        dt=1e-3,
        z=-0.2,
        inertia=np.array([1e-4]*3),
    )
    # Explicit would be -100; implicit gives -7.298437...
    assert force[3] == pytest.approx(-7.298437881283576, rel=1e-10)


def test_buoyancy_moment_not_double_counted():
    """D4: the total force already contains buoyancy; add r x F once."""
    force, _ = _run(
        linear_coefficient=0.0,
        use_buoyancy=True,
        buoyancy_center=(0.05, 0.0, 0.0),
    )
    assert force[2] == pytest.approx(4.905, rel=1e-12)
    assert force[4] == pytest.approx(-0.24525, rel=1e-12)


def test_buoyancy_magnitude_uses_link_fraction():
    """Single-fraction design: a CoB offset does not scale buoyancy."""
    force, _ = _run(
        linear_coefficient=0.0,
        use_buoyancy=True,
        buoyancy_center=(0.0, 0.0, 0.2),
    )
    assert force[2] == pytest.approx(4.905, rel=1e-12)


def test_water_properties_sampled_at_cob():
    """D5: water density/velocity use the true CoB height."""
    sampled = []

    def density(t, x, y, z):  # pylint: disable=unused-argument
        sampled.append(float(z))
        return 1000.0

    from farms_mujoco.swimming.drag import WaterPropertiesExtension
    water = WaterPropertiesExtension(
        surface=lambda t, x, y: 0.0,
        density=density,
        velocity=lambda t, x, y, z: np.array([0.0, 0.0, 0.0]),
        viscosity=lambda t, x, y, z: 1.0,
    )
    # 45 deg about x; CoB offset z=0.2 maps to global z=0.2*cos(45).
    quat = np.array([np.sin(np.pi/8), 0.0, 0.0, np.cos(np.pi/8)])
    _run(
        quat=quat,
        linear_coefficient=0.0,
        use_buoyancy=True,
        buoyancy_center=(0.0, 0.0, 0.2),
        water=water,
    )
    assert sampled == [pytest.approx(0.2*np.cos(np.pi/4), rel=1e-12)]


def test_drag_opposes_motion():
    """Negative coefficients with negative velocity produce positive drag."""
    force, _ = _run(z=-0.2, linear_velocity=-1.0)
    assert force[0] == pytest.approx(0.9804864072151531, rel=1e-10)


def test_zero_inertia_uses_explicit_torque():
    """A zero/invalid inertia falls back to explicit angular drag, not NaN."""
    force, _ = _run(
        linear_coefficient=0.0,
        angular_coefficient=-0.01,
        angular_velocity=100.0,
        dt=1e-3,
        z=-0.2,
        inertia=np.zeros(3),
    )
    assert np.all(np.isfinite(force))
    assert force[3] == pytest.approx(-100.0, rel=1e-12)


def test_drag_adds_to_existing_contribution():
    """Swimming adds to another extension's row instead of overwriting it."""
    initial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    force, _ = _run(z=-0.2, initial_force=initial)
    assert force[0] == pytest.approx(
        1.0 - 0.9804864072151531, rel=1e-10,
    )
