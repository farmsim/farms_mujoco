"""
Salamander swimming with SPH

The salamander model is spawned above water and attempts to move with a swimming
gait in the simulated water.
"""
from __future__ import print_function
import sys
import traceback
# import os
import numpy as np
import numpy

from pysph.base.utils import (
    get_particle_array,
    get_particle_array_wcsph,
    # get_particle_array_rigid_body
)
# PySPH base and carray imports
# from pysph.base.kernels import CubicSpline
from pysph.base.kernels import QuinticSpline
# from pysph.base.kernels import WendlandQuintic

from pysph.solver.solver import Solver
from pysph.sph.integrator import PECIntegrator  # EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (
    Equation,
    XSPHCorrection,
    ContinuityEquation
)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    # BodyForce, RigidBodyCollision,
    # RigidBodyMoments, RigidBodyMotion,
    # RK2StepRigidBody,
    IntegratorStep,
    AkinciRigidFluidCoupling,
)
from pysph.base.reduce_array import parallel_reduce_array
# from pysph.base.reduce_array import serial_reduce_array

# Rigid body physics
import pybullet
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation


def get_3d_dam(position, length=10, width=15, depth=10, dx=0.1, layers=2):
    _x = np.arange(0, length, dx)
    _y = np.arange(0, width, dx)
    _z = np.arange(0, depth, dx)

    x, y, z = np.meshgrid(_x, _y, _z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()

    # get particles inside the tank
    tmp = layers - 1
    cond_1 = (x > tmp * dx) & (x < _x[-1] - tmp * dx) & (z > tmp * dx)

    cond_2 = (y > tmp * dx) & (y < y[-1] - tmp * dx)

    cond = cond_1 & cond_2
    # exclude inside particles
    x, y, z = x[~cond], y[~cond], z[~cond]

    # Offset position
    x += position[0]
    y += position[1]
    z += position[2]

    return x, y, z


def get_3d_block(position, length=10, width=15, depth=10, dx=0.1):
    x = position[0] + np.arange(0, length, dx)
    y = position[1] + np.arange(0, width, dx)
    z = position[2] + np.arange(0, depth, dx)

    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    return x, y, z


def get_fluid_and_dam_geometry_3d(
        position,
        d_l, d_h, d_d, f_l, f_h, f_d, d_layers, d_dx,
        f_dx, fluid_left_extreme=None,
        tank_outside=False
):
    xd, yd, zd = get_3d_dam(position, d_l, d_h, d_d, d_dx, d_layers)
    xf, yf, zf = get_3d_block(position, f_l, f_h, f_d, f_dx)

    if fluid_left_extreme:
        x_trans, y_trans, z_trans = fluid_left_extreme
        xf += x_trans
        yf += y_trans
        zf += z_trans

    else:
        xf += d_layers * d_dx
        yf += d_layers * d_dx
        zf += d_layers * d_dx

    return xd, yd, zd, xf, yf, zf


# def get_sphere(centre=[0, 0, 0], radius=1, dx=0.1):
#     x = np.arange(0, radius * 2, dx)
#     y = np.arange(0, radius * 2, dx)
#     z = np.arange(0, radius * 2, dx)

#     x, y, z = np.meshgrid(x, y, z)
#     x, y, z = x.ravel(), y.ravel(), z.ravel()

#     cond = ((x - radius)**2 + (y - radius)**2) + (z - radius)**2 <= radius**2

#     x, y, z = x[cond], y[cond], z[cond]

#     x_trans = centre[0] - radius
#     y_trans = centre[1] - radius
#     z_trans = centre[2] - radius

#     x = x + x_trans
#     y = y + y_trans
#     z = z + z_trans

#     return x, y, z


def mesh_to_particles(link, filename=None, dx=2e-2, show=False):
    """Test"""
    import trimesh
    folder = "../farms_bullet/experiments/salamander/meshes/"
    mesh_file = "salamander_body_{}.obj".format(link)
    print("Loading {}".format(mesh_file))
    mesh = trimesh.load_mesh(folder+mesh_file)
    b = mesh.bounds
    d = np.array([np.arange(b[0, i], b[1, i], dx) for i in range(3)])
    points_tested = [[x, y, z] for x in d[0] for y in d[1] for z in d[2]]
    c = mesh.contains(points_tested)
    ind = np.where(c)[0]
    points = np.array([points_tested[i] for i in ind])
    if show:
        trimesh.Scene([trimesh.PointCloud(points)]).show()
    print("Number of particles in {}: {}".format(
        mesh_file,
        np.shape(points)[0]
    ))
    return points.T


def get_particle_array_rigid_body_bullet(constants=None, **props):
    """Return a particle array for a rigid body motion from Bullet."""

    extra_props = [
        'au', 'av', 'aw', 'V', 'fx', 'fy', 'fz', 'x0', 'y0', 'z0',
        'tang_disp_x', 'tang_disp_y', 'tang_disp_z', 'tang_disp_x0',
        'tang_disp_y0', 'tang_disp_z0', 'tang_velocity_x',
        'tang_velocity_y', 'rad_s',
        'tang_velocity_z',  'nx', 'ny', 'nz'
    ]

    body_id = props.pop('body_id', None)
    nb = 1 if body_id is None else numpy.max(body_id) + 1

    consts = {
        'num_body': numpy.asarray(nb, dtype=int),
        'cm': numpy.zeros(3*nb, dtype=float),
        'force': numpy.zeros(3*nb, dtype=float),
        'torque': numpy.zeros(3*nb, dtype=float),
    }
    if constants:
        consts.update(constants)
    pa = get_particle_array(
        constants=consts,
        additional_props=extra_props,
        **props
    )
    pa.add_property('body_id', type='int', data=body_id)
    pa.set_output_arrays([
        'x', 'y', 'z',
        'u', 'v', 'w',
        'rho', 'h', 'm',
        'p', 'pid',
        # 'au', 'av', 'aw',
        'tag', 'gid', 'V',
        'fx', 'fy', 'fz',
        'body_id'
    ])
    return pa


class BodyForceReset(Equation):
    """Body forces reset"""

    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        """Initializing"""
        d_fx[d_idx] = 0
        d_fy[d_idx] = 0
        d_fz[d_idx] = 0


class BulletPhysicsForces(Equation):

    def __init__(self, dest, sources, simulation, link_i):
        self.simulation = simulation
        self.link_i = link_i
        super(BulletPhysicsForces, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        # if dst.gpu:
        #     dst.gpu.pull('force', 'torque')
        iteration = self.simulation.iteration
        hydrodynamics = self.simulation.animat.data.sensors.hydrodynamics.array
        if self.simulation.iteration:
            # Compute force and torque in local frame
            ori = np.array(
                pybullet.getMatrixFromQuaternion(
                    self.simulation.animat.data.sensors.gps.urdf_orientation(
                        iteration-1,
                        self.link_i
                    )
                )
            ).reshape([3, 3])
            force = np.dot(
                ori.T,
                [dst.force[0], dst.force[1], dst.force[2]]
            )
            torque = np.dot(
                ori.T,
                [dst.torque[0], dst.torque[1], dst.torque[2]]
            )
            hydrodynamics[iteration, self.link_i, 0] = force[0]
            hydrodynamics[iteration, self.link_i, 1] = force[1]
            hydrodynamics[iteration, self.link_i, 2] = force[2]
            hydrodynamics[iteration, self.link_i, 3] = torque[0]
            hydrodynamics[iteration, self.link_i, 4] = torque[1]
            hydrodynamics[iteration, self.link_i, 5] = torque[2]


class BulletPhysicsUpdate(Equation):

    def __init__(self, dest, sources, simulation):
        self.simulation = simulation
        super(BulletPhysicsUpdate, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        self.simulation.step(self.simulation.iteration)


class BulletPhysicsMotion(Equation):

    def __init__(self, dest, sources, simulation, link_i):
        self.simulation = simulation
        self.link_i = link_i
        super(BulletPhysicsMotion, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        # if dst.gpu:
        #     dst.gpu.pull('x', 'y', 'z', 'u', 'v', 'w')  # , 'au', 'av', 'aw'
        position = np.array(
            self.simulation.animat.data.sensors.gps.urdf_position(
                self.simulation.iteration,
                self.link_i
            )
        )
        velocity = np.array(
            self.simulation.animat.data.sensors.gps.com_lin_velocity(
                self.simulation.iteration,
                self.link_i
            )
        )
        angular_velocity = np.array(
            self.simulation.animat.data.sensors.gps.com_ang_velocity(
                self.simulation.iteration,
                self.link_i
            )
        )
        orientation = np.array(
            self.simulation.animat.data.sensors.gps.urdf_orientation(
                self.simulation.iteration,
                self.link_i
            )
        )
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(
            orientation
        )).reshape([3, 3])
        # Position offset
        particles_p_local = np.array([
            np.dot(rot_matrix, particle)
            for particle in self.simulation.animat.particles[self.link_i]
        ])
        # Joint position + Particles positions
        dst.x = position[0] + particles_p_local[:, 0]
        dst.y = position[1] + particles_p_local[:, 1]
        dst.z = position[2] + particles_p_local[:, 2]
        # Particles velocities
        particle_v = np.array([
            velocity
            + np.cross(
                angular_velocity,
                np.dot(rot_matrix, particle)
            )
            for particle in self.simulation.animat.particles[self.link_i]
        ])
        dst.u, dst.v, dst.w = particle_v.T
        if dst.gpu:
            dst.gpu.push('x', 'y', 'z', 'u', 'v', 'w')  # , 'au', 'av', 'aw'


class RigidBodyMoments(Equation):

    def __init__(self, dest, sources, simulation, link_i):
        self.simulation = simulation
        self.link_i = link_i
        super(RigidBodyMoments, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        iteration = declare('int')
        simulation = declare('object')
        nbody = declare('int')
        i = declare('int')
        base_body = declare('int')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        cond = declare('object')
        simulation = self.simulation
        iteration = simulation.iteration
        if iteration:
            nbody = dst.num_body[0]
            if dst.gpu:
                dst.gpu.pull(
                    # 'omega',
                    'x', 'y', 'z',
                    'fx', 'fy', 'fz'
                )

            for i in range(nbody):
                cond = dst.body_id == i
                base_body = i*3

                # Position
                x = dst.x[cond]
                y = dst.y[cond]
                z = dst.z[cond]

                # the total force and torque
                fx = dst.fx[cond]
                fy = dst.fy[cond]
                fz = dst.fz[cond]
                dst.force[base_body + 0] = numpy.sum(fx)
                dst.force[base_body + 1] = numpy.sum(fy)
                dst.force[base_body + 2] = numpy.sum(fz)

                # Compute CoM
                cx, cy, cz = simulation.animat.data.sensors.gps.com_position(
                    iteration-1,
                    self.link_i
                )
                dst.cm[base_body + 0] = cx
                dst.cm[base_body + 1] = cy
                dst.cm[base_body + 2] = cz
                # Calculate the torque and reduce it.
                # Find torque about the Center of Mass and not origin.
                dst.torque[base_body + 0] = numpy.sum((y-cy)*fz - (z-cz)*fy)
                dst.torque[base_body + 1] = numpy.sum((z-cz)*fx - (x-cx)*fz)
                dst.torque[base_body + 2] = numpy.sum((x-cx)*fy - (y-cy)*fx)

            if dst.gpu:
                dst.gpu.push(
                    # 'total_mass',
                    # 'mi',
                    'cm',
                    'force',
                    # 'ac',
                    'torque',
                    # 'omega_dot'
                )


class BulletIterationUpdate(Equation):

    def __init__(self, dest, sources, simulation):
        self.simulation = simulation
        super(BulletIterationUpdate, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        self.simulation.iteration += 1


def declare(*args):
    """Dummy function"""
    pass


class RigidFluidCoupling(Application):
    """Rigid body to fluid coupling"""

    def __init__(self, **kwargs):

        # Simulation options
        output_dir = kwargs.pop("output_dir", None)

        ## Fluid dynamics options
        self.density_solid = kwargs.pop("density", 1000)
        # self.n_fluid_particles = kwargs.pop("n_fluid_particles", 8000)
        self.dt = 1e-3
        self.duration = 3

        # Pool
        self.tank_size = [4, 2, 0.1]  # [m]

        ## Rigid body physics

        # Animat options
        animat_options = SalamanderOptions(
            collect_gps=False,
            show_hydrodynamics=True,
            density=self.density_solid,
            scale=1
        )
        animat_options.morphology.density = [0, 0, self.tank_size[2]+0.05]
        animat_options.spawn.position = [0, 0, self.tank_size[2]+0.05]
        animat_options.control.drives.forward = kwargs.pop("drive", 4)
        animat_options.physics.viscous = False
        animat_options.physics.sph = True
        # Simulation options
        simulation_options = SimulationOptions.with_clargs(
            duration=self.duration,
            timestep=self.dt,
            fast=True,
            headless=True,
            log_path=output_dir+"/rigid"
        )
        simulation_options.units.meters = 1
        simulation_options.units.seconds = 1000
        simulation_options.units.kilograms = 1
        self.simulation = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
        )

        super(RigidFluidCoupling, self).__init__()
        if output_dir:
            self.args.append("--directory")
            self.args.append(output_dir)

    def initialize(self):
        self.tank_volume = self.tank_size[0]*self.tank_size[1]*self.tank_size[2]
        # self._spacing = 1e3*(self.tank_volume/self.n_fluid_particles)**(1/3)
        # print("Spacing: {} [mm]".format(self._spacing))
        # self.spacing = self._spacing * 1e-3  # [m]
        # self.dx = self.spacing
        self.dx = 2e-2
        self.hdx = 1.2
        self.ro = 1000
        self.solid_rho = 800
        self.m = 1000 * self.dx * self.dx * self.dx
        # The reference speed of sound, c0, is to be taken approximately as
        # 10 times the maximum expected velocity in the system. The particle
        # sound speed is given by the usual expression:
        # self.co = 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
        # self.co = 0.1  # 2 * np.sqrt(2 * 9.81 * 1000 * 1e-3)
        # self.co = 0.5*9.81  # 2 * np.sqrt(2 * 9.81 * 150 * 1e-3)
        gravity = 9.81
        falling_height = 0.05
        self.co = 10*np.sqrt(2*gravity*falling_height)
        # self.alpha = 0.1
        # self.alpha = 0.5  # Had to change it to avoid crashing
        # self.alpha = 0.5  # Had to change it to avoid crashing
        # # self.beta = 0.0  # Original
        # self.beta = 0.5

        # # Original used for salamander
        # self.alpha = 0.5  # 0.5  # Had to change it to avoid crashing
        # self.beta = 0.0  # 0.5
        # self.gamma = 7.0

        self.alpha = 1.0  # 0.5  # Had to change it to avoid crashing
        self.beta = 1.0  # 0.5
        self.gamma = 7.0

    def create_particles(self):

        # get coordinates of tank and fluid
        flu_len = self.tank_size[0]*1000
        flu_wid = self.tank_size[1]*1000
        flu_dep = self.tank_size[2]*1000

        layers = 2
        d_dx_ratio = 1
        tank_len = flu_len + d_dx_ratio * 2 * layers * 1e3*self.dx
        tank_wid = flu_wid + d_dx_ratio * 2 * layers * 1e3*self.dx
        tank_dep = flu_dep + d_dx_ratio * 2 * layers * 1e3*self.dx

        position = np.array([
            -tank_len + 1e3*1.1,
            -0.5*tank_wid,
            -d_dx_ratio * layers * 1e3*self.dx
        ])
        xt, yt, zt, xf, yf, zf = get_fluid_and_dam_geometry_3d(
            position=position,
            d_l=tank_len,
            d_h=tank_wid,
            d_d=tank_dep,
            f_l=flu_len,
            f_h=flu_wid,
            f_d=flu_dep,
            d_layers=layers,
            d_dx=d_dx_ratio*1e3*self.dx,
            f_dx=1e3*self.dx
        )
        # scale it to mm
        xt, yt, zt, xf, yf, zf = (
            xt * 1e-3,
            yt * 1e-3,
            zt * 1e-3,
            xf * 1e-3,
            yf * 1e-3,
            zf * 1e-3
        )

        # get coordinates of cube
        # xc, yc, zc = get_3d_block(100, 100, 100, 1e3*self.dx/2.)
        # xc, yc, zc = (
        #     (xc + 200) * 1e-3 + d_dx_ratio * layers * self.dx,
        #     (yc + 550) * 1e-3 + d_dx_ratio * layers * self.dx,
        #     (zc + 200) * 1e-3 + d_dx_ratio * layers * self.dx
        # )
        # link_positions = [  # From SDF
        #     [0, 0, 0],
        #     [0.200000003, 0, 0.0069946074],
        #     [0.2700000107, 0, 0.010382493],
        #     [0.3400000036, 0, 0.0106022889],
        #     [0.4099999964, 0, 0.010412137],
        #     [0.4799999893, 0, 0.0086611426],
        #     [0.5500000119, 0, 0.0043904358],
        #     [0.6200000048, 0, 0.0006898994],
        #     [0.6899999976, 0, 8.0787e-06],
        #     [0.7599999905, 0, -4.89001e-05],
        #     [0.8299999833, 0, 0.0001386079],
        #     [0.8999999762, 0, 0.0003494423]
        # ]
        link_positions = np.zeros([12, 3])
        xc = [None for _ in range(12)]
        yc = [None for _ in range(12)]
        zc = [None for _ in range(12)]
        for i in range(12):
            xc[i], yc[i], zc[i] = mesh_to_particles(i, dx=self.dx/2)
        self.simulation.animat.particles = np.array([
            [
                [xi, yi, zi]
                for xi, yi, zi in zip(x, y, z)
            ]
            for x, y, z in zip(xc, yc, zc)
        ])

        # Create particle array for fluid
        m = self.ro * self.dx * self.dx * self.dx
        rho = self.ro
        h = self.hdx * self.dx
        fluid = get_particle_array_wcsph(
            x=xf, y=yf, z=zf, h=h,
            m=m, rho=rho, name="fluid"
        )

        # Create particle array for tank
        m = 1000 * self.dx**3
        rho = 1000
        rad_s = self.dx / 2.
        h = self.hdx * self.dx
        V = self.dx**3
        tank = get_particle_array_wcsph(
            x=xt, y=yt, z=zt, h=h,
            m=m, rho=rho, rad_s=rad_s, V=V, name="tank"
        )
        for name in ['fx', 'fy', 'fz']:
            tank.add_property(name)

        # Links
        cube = [None for _ in range(12)]

        # Create particle array for cube
        h = self.hdx * self.dx/2.

        # assign  density of three spheres
        rho = [np.ones_like(_xc) * self.density_solid for _xc in xc]

        # assign body id's
        body = [np.ones_like(_xc, dtype=int) * 0 for _xc in xc]

        m = [_rho * (self.dx/2.)**3 for _rho in rho]
        rad_s = self.dx / 4.
        V = (self.dx/2.)**3
        cs = 0.0
        cube = [
            get_particle_array_rigid_body_bullet(
                x=xc[i], y=yc[i], z=zc[i], h=h,
                m=m[i], rho=rho[i],
                rad_s=rad_s,
                V=V,
                cs=cs,
                body_id=body[i],
                name="cube_{}".format(i)
            )
            for i in range(12)
        ]

        print("Number of fluid particles: {}".format(
            fluid.get_number_of_particles()
        ))
        print("Number of tank particles: {}".format(
            tank.get_number_of_particles()
        ))
        print("Number of salamander particles: {}".format(
            [cube[i].get_number_of_particles() for i in range(12)]
        ))
        return [fluid, tank] + cube

    def create_solver(self):
        # kernel = CubicSpline(dim=3)
        kernel = QuinticSpline(dim=3)
        # kernel = WendlandQuintic(dim=3)

        # simulation = Simulation()
        # cubes = {
        #     "cube_{}".format(i): RK2StepRigidBody(simulation=self.simulation)
        #     for i in range(12)
        # }
        # integrator = EPECIntegrator(
        #     fluid=WCSPHStep(),
        #     tank=WCSPHStep(),
        #     # **cubes
        # )
        integrator = PECIntegrator(
            fluid=WCSPHStep(),
            tank=WCSPHStep(),
            # **cubes
        )

        # dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        print("DT: %s" % self.dt)
        # tf = 1.2  # 5
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=self.dt,
            tf=self.duration - self.dt,
            adaptive_timestep=False,  # False because otherwise not constant
            reorder_freq=0
        )
        solver.set_output_fname("salamander")  # Does not work
        return solver

    def create_equations(self):

        # Reset forces
        rigid_force_reset = [
            Group(
                equations=[
                    BodyForceReset(
                        dest='cube_{}'.format(i),
                        sources=None
                    )
                    for i in range(12)
                ]
            )
        ]

        # Compute fluid motion
        continuity = [
            Group(
                equations=[
                    ContinuityEquation(
                        dest='fluid',
                        sources=['fluid', 'tank'] + ['cube_{}'.format(i) for i in range(12)]
                    ),
                    ContinuityEquation(
                        dest='tank',
                        sources=['tank', 'fluid'] + ['cube_{}'.format(i) for i in range(12)]
                    )
                ]
            )
        ]
        tait = [
            # Tait equation of state
            Group(
                equations=[
                    TaitEOSHGCorrection(
                        dest='fluid',
                        sources=None,
                        rho0=self.ro,
                        c0=self.co,
                        gamma=self.gamma
                    ),
                    TaitEOSHGCorrection(
                        dest='tank',
                        sources=None,
                        rho0=self.ro,
                        c0=self.co,
                        gamma=self.gamma
                    ),
                ],
                real=False
            )
        ]
        fluid_motion = [
            Group(
                equations=[
                    MomentumEquation(
                        dest='fluid',
                        sources=['fluid', 'tank'],
                        alpha=self.alpha,
                        beta=self.beta,
                        c0=self.co,
                        gz=-9.81
                    )
                ] + [
                    AkinciRigidFluidCoupling(
                        dest='fluid',
                        sources=['cube_{}'.format(i)]
                    )
                    for i in range(12)
                ] + [
                    XSPHCorrection(
                        dest='fluid',
                        sources=['fluid', 'tank']
                    ),
                ]
            )
        ]

        # Compute forces and torques applied to rigid bodies
        cube_moment = [
            Group(
                equations=[
                    RigidBodyMoments(
                        dest='cube_{}'.format(i),
                        sources=None,
                        simulation=self.simulation,
                        link_i=i
                    )
                    for i in range(12)
                ]
            )
        ]

        # Send forces to Bullet physics from particles information
        bullet_forces = [
            Group(
                equations=[
                    BulletPhysicsForces(
                        dest='cube_{}'.format(i),
                        sources=None,
                        simulation=self.simulation,
                        link_i=i
                    )
                    for i in range(12)
                ]
            )
        ]

        # Update motion in Bullet
        bullet_update = [
            Group(
                equations=[
                    BulletPhysicsUpdate(
                        dest='cube_0',
                        sources=None,
                        simulation=self.simulation
                    )
                ]
            )
        ]

        # Update particles from Bullet physics information
        bullet_motion = [
            Group(
                equations=[
                    BulletPhysicsMotion(
                        dest='cube_{}'.format(i),
                        sources=None,
                        simulation=self.simulation,
                        link_i=i
                    )
                    for i in range(12)
                ]
            )
        ]

        # Update iteration
        iteration_update = [
            Group(
                equations=[
                    BulletIterationUpdate(
                        dest='cube_0',
                        sources=None,
                        simulation=self.simulation
                    )
                ]
            )
        ]

        equations = (
            rigid_force_reset
            + continuity
            + tait
            + fluid_motion
            + cube_moment
            + bullet_forces
            + bullet_update
            + bullet_motion
            + iteration_update
        )
        return equations


def main():
    """Main"""
    # for density in [500, 800, 900, 1000, 1100, 1200, 2000]:
    # for density in [1000]:
    #     for drive in [4]:
    for density in [1000]:  # 300, 500, 700, 1000
        # for drive in [3.001, 3.5, 4, 4.5, 4.999]:
        for drive in [4]:
            try:
                print("Density: {}".format(density))
                app = RigidFluidCoupling(
                    density=density,
                    drive=drive,
                    output_dir="benchmark_swimming_density{}_drive{}".format(
                        density,
                        drive
                    ).replace(".", "p")
                )
                app.run()
                app.simulation.postprocess(
                    iteration=app.simulation.iteration,
                    log_path=app.simulation.options.log_path,
                    log_extension=app.simulation.options.log_extension
                )
                app.simulation.end()
            except Exception as err:
                print("WATCH OUT THERE WAS AN ERROR!!!")
                traceback.print_exc(file=sys.stdout)
                print(str(err))


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)



def debug():
    """Debug"""
    particles = mesh_to_particles(9, dx=1e-2, show=True)
    for i, ori in enumerate(["x", "y", "z"]):
        print("{0}Min: {1} {0}Max: {2}".format(
            ori,
            min(particles[i, :]),
            max(particles[i, :])
        ))
    print(get_3d_block([0, 0, 0]))


if __name__ == '__main__':
    main()
    # profile()
