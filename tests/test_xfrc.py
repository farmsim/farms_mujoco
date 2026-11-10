"""Regression tests for the shared xfrc accumulator."""

from types import SimpleNamespace

import numpy as np

from farms_mujoco.simulation.physics import physics2data
from farms_mujoco.simulation.task import ExperimentTask


def test_physics2data_resets_only_current_xfrc_row(monkeypatch):
    """Sensor collection clears the current row, not the other slots."""
    import farms_mujoco.simulation.physics as physics_module

    monkeypatch.setattr(
        physics_module, 'physicslinks2data', lambda *a, **k: None,
    )
    monkeypatch.setattr(
        physics_module, 'physicslinksvelsensors2data', lambda *a, **k: None,
    )
    xfrc = np.full((4, 2, 6), 7.0)
    data = SimpleNamespace(
        sensors=SimpleNamespace(
            xfrc=SimpleNamespace(array=xfrc),
        ),
    )
    physics2data(
        physics=None,
        iteration=2,
        data=data,
        maps={'sensors': {}},
        units=SimpleNamespace(),
        links_only=True,
    )
    assert np.all(xfrc[2] == 0.0)
    assert np.all(xfrc[1] == 7.0)
    assert np.all(xfrc[3] == 7.0)


def _apply_task(xfrc_row, newtons=1.0, torques=1.0, prior=None):
    """Run ExperimentTask.apply_xfrc with two identity-oriented bodies."""
    task = object.__new__(ExperimentTask)
    task.iteration = 0
    task.buffer_size = 10
    task.units = SimpleNamespace(newtons=newtons, torques=torques)
    task.data = SimpleNamespace(animats=[SimpleNamespace(
        sensors=SimpleNamespace(xfrc=SimpleNamespace(
            array=np.array([xfrc_row], dtype=float),
        )),
    )])
    task.maps = [{'sensors': {'data2xfrc': np.array([0, 1])}}]
    # Two bodies, identity rotation; only body 0 receives data.
    xmat = np.zeros((2, 9))
    xmat[:, [0, 4, 8]] = 1.0
    applied = np.zeros((2, 6)) if prior is None else prior.copy()
    physics = SimpleNamespace(data=SimpleNamespace(
        xmat=xmat,
        xfrc_applied=applied,
    ))
    ExperimentTask.apply_xfrc(task, physics)
    return physics.data.xfrc_applied


def test_apply_xfrc_scales_and_replaces():
    """The applier scales units and overwrites, so forces do not accumulate."""
    local = np.array([
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    applied = _apply_task(
        local, newtons=2.0, torques=3.0,
        prior=np.full((2, 6), 99.0),
    )
    assert np.allclose(applied[0], [2, 4, 6, 12, 15, 18])
    assert np.allclose(applied[1], 0.0)


def test_apply_xfrc_is_order_independent_for_contributions():
    """Two contributions give the same applied force in either order."""
    a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 1.0])
    zeros = np.zeros(6)
    row_ab = np.stack([a + b, zeros])
    row_ba = np.stack([b + a, zeros])
    applied_ab = _apply_task(row_ab)
    applied_ba = _apply_task(row_ba)
    assert np.allclose(applied_ab, applied_ba)
