"""3D benchmark with rigid boy physics coupling

A cube with a specified density falls into a pool of water. This is based on the
three_cubes_in_vessel_3d.py example provided by PySPH.

"""
from __future__ import print_function
import sys
# import os
import numpy as np
import numpy

from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
# PySPH base and carray imports
from pysph.base.kernels import CubicSpline
# from pysph.base.kernels import WendlandQuintic

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
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

# Rigid body physics
import pybullet
from farms_bullet.experiments.salamander.animat_options import SalamanderOptions
from farms_bullet.simulations.simulation_options import SimulationOptions
from farms_bullet.experiments.salamander.simulation import SalamanderSimulation


def get_3d_dam(length=10, width=15, depth=10, dx=0.1, layers=2):
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

    return x, y, z


def get_3d_block(length=10, width=15, depth=10, dx=0.1):
    x = np.arange(0, length, dx)
    y = np.arange(0, width, dx)
    z = np.arange(0, depth, dx)

    x, y, z = np.meshgrid(x, y, z)
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    return x, y, z


def get_fluid_and_dam_geometry_3d(d_l, d_h, d_d, f_l, f_h, f_d, d_layers, d_dx,
                                  f_dx, fluid_left_extreme=None,
                                  tank_outside=False):
    xd, yd, zd = get_3d_dam(d_l, d_h, d_d, d_dx, d_layers)
    xf, yf, zf = get_3d_block(f_l, f_h, f_d, f_dx)

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


class BulletPhysicsForces(Equation):

    def __init__(self, dest, sources, simulation, link_i):
        self.simulation = simulation
        self.link_i = link_i
        super(BulletPhysicsForces, self).__init__(dest, sources)

    def py_initialize(self, dst, t, dt):
        if dst.gpu:
            dst.gpu.pull('force', 'torque')
        iteration = self.simulation.iteration
        hydrodynamics = self.simulation.animat.data.sensors.hydrodynamics.array
        hydrodynamics[iteration, self.link_i, 0] = dst.force[0]
        hydrodynamics[iteration, self.link_i, 1] = dst.force[1]
        hydrodynamics[iteration, self.link_i, 2] = dst.force[2]
        hydrodynamics[iteration, self.link_i, 3] = dst.torque[0]
        hydrodynamics[iteration, self.link_i, 4] = dst.torque[1]
        hydrodynamics[iteration, self.link_i, 5] = dst.torque[2]


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
        if dst.gpu:
            dst.gpu.pull('x', 'y', 'z', 'u', 'v', 'w', 'au', 'av', 'aw')
        # base = d_body_id[d_idx]*3
        # wx = dst.omega[base + 0]
        # wy = dst.omega[base + 1]
        # wz = dst.omega[base + 2]
        # rx = dst.x[d_idx] - dst.cm[base + 0]
        # ry = dst.y[d_idx] - dst.cm[base + 1]
        # rz = dst.z[d_idx] - dst.cm[base + 2]
        # dst.u[d_idx] = dst.vc[base + 0] + wy*rz - wz*ry
        # dst.v[d_idx] = dst.vc[base + 1] + wz*rx - wx*rz
        # dst.w[d_idx] = dst.vc[base + 2] + wx*ry - wy*rx
        # print("Positions for link {}:".format(self.link_i))
        # print(dst.x)
        # print(self.simulation.animat.particles[self.link_i][:, 0])
        position = np.array(
            self.simulation.animat.data.sensors.gps.urdf_position(
                self.simulation.iteration,
                self.link_i
            )
        )
        if self.link_i == 11:
            print("SPH position: {}".format(np.array(position)))
        orientation = np.array(
            self.simulation.animat.data.sensors.gps.urdf_orientation(
                self.simulation.iteration,
                self.link_i
            )
        )
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(
            orientation
        )).reshape([3, 3])
        particles = np.array([
            np.dot(rot_matrix, np.array(particle))
            for particle in self.simulation.animat.particles[self.link_i]
        ])
        dst.x, dst.y, dst.z = [
            # Joint position
            position[i]
            # Particles positions
            + particles[:, i]
            for i in range(3)
        ]
        size = len(self.simulation.animat.particles[self.link_i])
        dst.u, dst.v, dst.w, dst.au, dst.av, dst.aw = np.zeros([6, size])
        if dst.gpu:
            dst.gpu.push('x', 'y', 'z', 'u', 'v', 'w', 'au', 'av', 'aw')


# class BodyForce(Equation):

#     def __init__(self, dest, sources, simulation, gx=0.0, gy=0.0, gz=0.0):
#         # self.gx = gx
#         # self.gy = gy
#         # self.gz = gz
#         self.simulation = simulation
#         super(BodyForce, self).__init__(dest, sources)

#     def py_initialize(self, dst, t, dt):
#         # Called once per destination array before initialize.
#         # This is a pure Python function and is not translated.
#         # print("Initializing body forces ({})".format(self.simulation.iteration))
#         self.simulation.iteration += 1
#         n_particles = dst.get_number_of_particles()
#         if dst.gpu:
#             dst.gpu.pull('fx', 'fy', 'fz')
#         for i in range(n_particles):
#             dst.fx[i] = 0
#             dst.fy[i] = 0.
#             dst.fz[i] = -9.81
#         if dst.gpu:
#             dst.gpu.push('fx', 'fy', 'fz')

#     def initialize(self, d_idx, d_m, d_fx, d_fy, d_fz):
#         # d_fx[d_idx] = d_m[d_idx]*self.gx
#         # d_fy[d_idx] = d_m[d_idx]*self.gy
#         # d_fz[d_idx] = d_m[d_idx]*self.gz
#         d_fx[d_idx] *= d_m[d_idx]
#         d_fy[d_idx] *= d_m[d_idx]
#         d_fz[d_idx] *= d_m[d_idx]


class RigidBodyMoments(Equation):
    def reduce(self, dst, t, dt):
        # FIXME: this will be slow in opencl
        nbody = declare('int')
        i = declare('int')
        base_mi = declare('int')
        base = declare('int')
        nbody = dst.num_body[0]
        if dst.gpu:
            dst.gpu.pull('omega', 'x', 'y', 'z', 'fx', 'fy', 'fz')

        d_mi = declare('object')
        m = declare('object')
        x = declare('object')
        y = declare('object')
        z = declare('object')
        fx = declare('object')
        fy = declare('object')
        fz = declare('object')
        d_mi = dst.mi
        cond = declare('object')
        for i in range(nbody):
            cond = dst.body_id == i
            base = i*16
            m = dst.m[cond]
            x = dst.x[cond]
            y = dst.y[cond]
            z = dst.z[cond]
            # Find the total_mass, center of mass and second moments.
            d_mi[base + 0] = numpy.sum(m)
            d_mi[base + 1] = numpy.sum(m*x)
            d_mi[base + 2] = numpy.sum(m*y)
            d_mi[base + 3] = numpy.sum(m*z)
            # Only do the lower triangle of values moments of inertia.
            d_mi[base + 4] = numpy.sum(m*(y*y + z*z))
            d_mi[base + 5] = numpy.sum(m*(x*x + z*z))
            d_mi[base + 6] = numpy.sum(m*(x*x + y*y))

            d_mi[base + 7] = -numpy.sum(m*x*y)
            d_mi[base + 8] = -numpy.sum(m*x*z)
            d_mi[base + 9] = -numpy.sum(m*y*z)

            # the total force and torque
            fx = dst.fx[cond]
            fy = dst.fy[cond]
            fz = dst.fz[cond]
            d_mi[base + 10] = numpy.sum(fx)
            d_mi[base + 11] = numpy.sum(fy)
            d_mi[base + 12] = numpy.sum(fz)

            # Calculate the torque and reduce it.
            d_mi[base + 13] = numpy.sum(y*fz - z*fy)
            d_mi[base + 14] = numpy.sum(z*fx - x*fz)
            d_mi[base + 15] = numpy.sum(x*fy - y*fx)

        # Reduce the temporary mi values in parallel across processors.
        d_mi[:] = parallel_reduce_array(dst.mi)

        # Set the reduced values.
        for i in range(nbody):
            base_mi = i*16
            base = i*3
            m = d_mi[base_mi + 0]
            dst.total_mass[i] = m
            cx = d_mi[base_mi + 1]/m
            cy = d_mi[base_mi + 2]/m
            cz = d_mi[base_mi + 3]/m
            dst.cm[base + 0] = cx
            dst.cm[base + 1] = cy
            dst.cm[base + 2] = cz

            # The actual moment of inertia about center of mass from parallel
            # axes theorem.
            ixx = d_mi[base_mi + 4] - (cy*cy + cz*cz)*m
            iyy = d_mi[base_mi + 5] - (cx*cx + cz*cz)*m
            izz = d_mi[base_mi + 6] - (cx*cx + cy*cy)*m
            ixy = d_mi[base_mi + 7] + cx*cy*m
            ixz = d_mi[base_mi + 8] + cx*cz*m
            iyz = d_mi[base_mi + 9] + cy*cz*m

            d_mi[base_mi + 0] = ixx
            d_mi[base_mi + 1] = ixy
            d_mi[base_mi + 2] = ixz
            d_mi[base_mi + 3] = ixy
            d_mi[base_mi + 4] = iyy
            d_mi[base_mi + 5] = iyz
            d_mi[base_mi + 6] = ixz
            d_mi[base_mi + 7] = iyz
            d_mi[base_mi + 8] = izz

            fx = d_mi[base_mi + 10]
            fy = d_mi[base_mi + 11]
            fz = d_mi[base_mi + 12]
            dst.force[base + 0] = fx
            dst.force[base + 1] = fy
            dst.force[base + 2] = fz

            # Acceleration of CM.
            dst.ac[base + 0] = fx/m
            dst.ac[base + 1] = fy/m
            dst.ac[base + 2] = fz/m

            # Find torque about the Center of Mass and not origin.
            tx = d_mi[base_mi + 13]
            ty = d_mi[base_mi + 14]
            tz = d_mi[base_mi + 15]
            tx -= cy*fz - cz*fy
            ty -= -cx*fz + cz*fx
            tz -= cx*fy - cy*fx
            dst.torque[base + 0] = tx
            dst.torque[base + 1] = ty
            dst.torque[base + 2] = tz

            wx = dst.omega[base + 0]
            wy = dst.omega[base + 1]
            wz = dst.omega[base + 2]
            # Find omega_dot from: omega_dot = inv(I) (\tau - w x (Iw))
            # This was done using the sympy code above.
            tmp0 = iyz**2
            tmp1 = ixy**2
            tmp2 = ixz**2
            tmp3 = ixx*iyy
            tmp4 = ixy*ixz
            tmp5 = 1./(ixx*tmp0 + iyy*tmp2 - 2*iyz*tmp4 + izz*tmp1 - izz*tmp3)
            tmp6 = ixy*izz - ixz*iyz
            tmp7 = ixz*wx + iyz*wy + izz*wz
            tmp8 = ixx*wx + ixy*wy + ixz*wz
            tmp9 = tmp7*wx - tmp8*wz + ty
            tmp10 = ixy*iyz - ixz*iyy
            tmp11 = ixy*wx + iyy*wy + iyz*wz
            tmp12 = -tmp11*wx + tmp8*wy + tz
            tmp13 = tmp11*wz - tmp7*wy + tx
            tmp14 = ixx*iyz - tmp4
            dst.omega_dot[base + 0] = tmp5*(-tmp10*tmp12 -
                                            tmp13*(iyy*izz - tmp0) + tmp6*tmp9)
            dst.omega_dot[base + 1] = tmp5*(tmp12*tmp14 +
                                            tmp13*tmp6 - tmp9*(ixx*izz - tmp2))
            dst.omega_dot[base + 2] = tmp5*(-tmp10*tmp13 -
                                            tmp12*(-tmp1 + tmp3) + tmp14*tmp9)
        if dst.gpu:
            dst.gpu.push(
                'total_mass', 'mi', 'cm', 'force', 'ac', 'torque',
                'omega_dot'
            )


# class RigidBodyCollision(Equation):
#     """Force between two spheres is implemented using DEM contact force law.

#     Refer https://doi.org/10.1016/j.powtec.2011.09.019 for more
#     information.

#     Open-source MFIX-DEM software for gas–solids flows:
#     Part I—Verification studies .

#     """
#     def __init__(self, dest, sources, kn=1e3, mu=0.5, en=0.8):
#         """Initialise the required coefficients for force calculation.


#         Keyword arguments:
#         kn -- Normal spring stiffness (default 1e3)
#         mu -- friction coefficient (default 0.5)
#         en -- coefficient of restitution (0.8)

#         Given these coefficients, tangential spring stiffness, normal and
#         tangential damping coefficient are calculated by default.

#         """
#         self.kn = kn
#         self.kt = 2. / 7. * kn
#         m_eff = np.pi * 0.5**2 * 1e-6 * 2120
#         self.gamma_n = -(2 * np.sqrt(kn * m_eff) * np.log(en)) / (
#             np.sqrt(np.pi**2 + np.log(en)**2))
#         self.gamma_t = 0.5 * self.gamma_n
#         self.mu = mu
#         super(RigidBodyCollision, self).__init__(dest, sources)

#     def loop(self, d_idx, d_fx, d_fy, d_fz, d_h, d_total_mass, d_rad_s,
#              d_tang_disp_x, d_tang_disp_y, d_tang_disp_z, d_tang_velocity_x,
#              d_tang_velocity_y, d_tang_velocity_z, s_idx, s_rad_s, XIJ, RIJ,
#              R2IJ, VIJ):
#         overlap = 0
#         if RIJ > 1e-9:
#             overlap = d_rad_s[d_idx] + s_rad_s[s_idx] - RIJ

#         if overlap > 0:
#             # normal vector passing from particle i to j
#             nij_x = -XIJ[0] / RIJ
#             nij_y = -XIJ[1] / RIJ
#             nij_z = -XIJ[2] / RIJ

#             # overlap speed: a scalar
#             vijdotnij = VIJ[0] * nij_x + VIJ[1] * nij_y + VIJ[2] * nij_z

#             # normal velocity
#             vijn_x = vijdotnij * nij_x
#             vijn_y = vijdotnij * nij_y
#             vijn_z = vijdotnij * nij_z

#             # normal force with conservative and dissipation part
#             fn_x = -self.kn * overlap * nij_x - self.gamma_n * vijn_x
#             fn_y = -self.kn * overlap * nij_y - self.gamma_n * vijn_y
#             fn_z = -self.kn * overlap * nij_z - self.gamma_n * vijn_z

#             # ----------------------Tangential force---------------------- #

#             # tangential velocity
#             d_tang_velocity_x[d_idx] = VIJ[0] - vijn_x
#             d_tang_velocity_y[d_idx] = VIJ[1] - vijn_y
#             d_tang_velocity_z[d_idx] = VIJ[2] - vijn_z

#             dtvx = d_tang_velocity_x[d_idx]
#             dtvy = d_tang_velocity_y[d_idx]
#             dtvz = d_tang_velocity_z[d_idx]
#             _tang = sqrt(dtvx*dtvx + dtvy*dtvy + dtvz*dtvz)

#             # tangential unit vector
#             tij_x = 0
#             tij_y = 0
#             tij_z = 0
#             if _tang > 0:
#                 tij_x = d_tang_velocity_x[d_idx] / _tang
#                 tij_y = d_tang_velocity_y[d_idx] / _tang
#                 tij_z = d_tang_velocity_z[d_idx] / _tang

#             # damping force or dissipation
#             ft_x_d = -self.gamma_t * d_tang_velocity_x[d_idx]
#             ft_y_d = -self.gamma_t * d_tang_velocity_y[d_idx]
#             ft_z_d = -self.gamma_t * d_tang_velocity_z[d_idx]

#             # tangential spring force
#             ft_x_s = -self.kt * d_tang_disp_x[d_idx]
#             ft_y_s = -self.kt * d_tang_disp_y[d_idx]
#             ft_z_s = -self.kt * d_tang_disp_z[d_idx]

#             ft_x = ft_x_d + ft_x_s
#             ft_y = ft_y_d + ft_y_s
#             ft_z = ft_z_d + ft_z_s

#             # coulomb law
#             ftij = sqrt((ft_x**2) + (ft_y**2) + (ft_z**2))
#             fnij = sqrt((fn_x**2) + (fn_y**2) + (fn_z**2))

#             _fnij = self.mu * fnij

#             if _fnij < ftij:
#                 ft_x = -_fnij * tij_x
#                 ft_y = -_fnij * tij_y
#                 ft_z = -_fnij * tij_z

#             d_fx[d_idx] += fn_x + ft_x
#             d_fy[d_idx] += fn_y + ft_y
#             d_fz[d_idx] += fn_z + ft_z
#         else:
#             d_tang_velocity_x[d_idx] = 0
#             d_tang_velocity_y[d_idx] = 0
#             d_tang_velocity_z[d_idx] = 0

#             d_tang_disp_x[d_idx] = 0
#             d_tang_disp_y[d_idx] = 0
#             d_tang_disp_z[d_idx] = 0


# class RigidBodyMotion(Equation):
#     def initialize(self, d_idx, d_x, d_y, d_z, d_u, d_v, d_w,
#                    d_cm, d_vc, d_ac, d_omega, d_body_id):
#         base = declare('int')
#         base = d_body_id[d_idx]*3
#         wx = d_omega[base + 0]
#         wy = d_omega[base + 1]
#         wz = d_omega[base + 2]
#         rx = d_x[d_idx] - d_cm[base + 0]
#         ry = d_y[d_idx] - d_cm[base + 1]
#         rz = d_z[d_idx] - d_cm[base + 2]

#         d_u[d_idx] = d_vc[base + 0] + wy*rz - wz*ry
#         d_v[d_idx] = d_vc[base + 1] + wz*rx - wx*rz
#         d_w[d_idx] = d_vc[base + 2] + wx*ry - wy*rx


# class RK2StepRigidBody(IntegratorStep):

#     def __init__(self, simulation):
#         super(RK2StepRigidBody, self).__init__()
#         self.simulation = simulation

#     def py_initialize(self, dst, t, dt):
#         # Called once per destination array before initialize.
#         # This is a pure Python function and is not translated.
#         # print("Running body step ({})".format(self.simulation.iteration))
#         self.simulation.iteration += 1
#         # n_particles = dst.get_number_of_particles()
#         if dst.gpu:
#             dst.gpu.pull('fx', 'fy', 'fz')
#         # for i in range(n_particles):
#         #     dst.fx[i] = -9.81
#         #     dst.fy[i] = 0.
#         #     dst.fz[i] = -9.81
#         if dst.gpu:
#             dst.gpu.push('fx', 'fy', 'fz')

#     def initialize(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_z0,
#                    d_omega, d_omega0, d_vc, d_vc0, d_num_body):
#         _i = declare('int')
#         _j = declare('int')
#         base = declare('int')
#         if d_idx == 0:
#             for _i in range(d_num_body[0]):
#                 base = 3*_i
#                 for _j in range(3):
#                     d_vc0[base + _j] = d_vc[base + _j]
#                     d_omega0[base + _j] = d_omega[base + _j]

#         d_x0[d_idx] = d_x[d_idx]
#         d_y0[d_idx] = d_y[d_idx]
#         d_z0[d_idx] = d_z[d_idx]

#     def stage1(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
#                d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
#                dt=0.0):
#         dtb2 = 0.5*dt
#         _i = declare('int')
#         j = declare('int')
#         base = declare('int')
#         if d_idx == 0:
#             for _i in range(d_num_body[0]):
#                 base = 3*_i
#                 for j in range(3):
#                     d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dtb2
#                     d_omega[base + j] = (d_omega0[base + j] +
#                                          d_omega_dot[base + j]*dtb2)

#         d_x[d_idx] = d_x0[d_idx] + dtb2*d_u[d_idx]
#         d_y[d_idx] = d_y0[d_idx] + dtb2*d_v[d_idx]
#         d_z[d_idx] = d_z0[d_idx] + dtb2*d_w[d_idx]

#     def stage2(self, d_idx, d_u, d_v, d_w, d_x, d_y, d_z, d_x0, d_y0, d_z0,
#                d_omega, d_omega_dot, d_vc, d_ac, d_omega0, d_vc0, d_num_body,
#                dt=0.0):
#         _i = declare('int')
#         j = declare('int')
#         base = declare('int')
#         if d_idx == 0:
#             for _i in range(d_num_body[0]):
#                 base = 3*_i
#                 for j in range(3):
#                     d_vc[base + j] = d_vc0[base + j] + d_ac[base + j]*dt
#                     d_omega[base + j] = (d_omega0[base + j] +
#                                          d_omega_dot[base + j]*dt)

#         d_x[d_idx] = d_x0[d_idx] + dt*d_u[d_idx]
#         d_y[d_idx] = d_y0[d_idx] + dt*d_v[d_idx]
#         d_z[d_idx] = d_z0[d_idx] + dt*d_w[d_idx]


def declare(*args):
    """Dummy function"""
    pass


class RigidFluidCoupling(Application):
    """Rigid body to fluid coupling"""

    def __init__(self, density=500, n_fluid_particles=8000, output_dir=None):

        ## Fluid dynamics options
        self.density = density
        self.n_fluid_particles = n_fluid_particles
        super(RigidFluidCoupling, self).__init__()
        self.dt = 1e-3
        if output_dir:
            self.args.append("--directory")
            self.args.append(output_dir)

        ## Rigid body physics

        # Animat options
        animat_options = SalamanderOptions(
            collect_gps=False,
            show_hydrodynamics=True,
            scale=1
        )
        animat_options.control.drives.forward = 4
        # Simulation options
        simulation_options = SimulationOptions.with_clargs()
        simulation_options.timestep = self.dt
        simulation_options.headless = False
        simulation_options.units.meters = 1
        simulation_options.units.seconds = 1
        simulation_options.units.kilograms = 1
        self.simulation = SalamanderSimulation(
            simulation_options=simulation_options,
            animat_options=animat_options
        )
        # self.simulation.step(self.simulation.iteration)
        # self.simulation.iteration += 1

    def initialize(self):
        self.density_cube = self.density
        self.tank_size = [1.2, 0.3, 0.2]  # [m]
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
        self.alpha = 0.5  # 0.5  # Had to change it to avoid crashing
        # self.beta = 0.0  # Original
        self.beta = 0.0  # 0.5
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

        xt, yt, zt, xf, yf, zf = get_fluid_and_dam_geometry_3d(
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
            xc[i], yc[i], zc[i] = (
                (
                    (xc[i] + self.tank_size[0] - 1.1 + link_positions[i][0])
                    + d_dx_ratio * layers * self.dx
                ),
                (
                    (yc[i] + self.tank_size[1]/2)
                    + d_dx_ratio * layers * self.dx
                ),
                (
                    (zc[i] +  + self.tank_size[2] + 0.03 + link_positions[i][2])
                    + d_dx_ratio * layers * self.dx
                )
            )
        self.simulation.animat.particles = [
            np.array([
                [xi, yi, zi]
                for xi, yi, zi in zip(x, y, z)
            ])
            for x, y, z in zip(xc, yc, zc)
        ]

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
        rho = [np.ones_like(_xc) * self.density_cube for _xc in xc]

        # assign body id's
        body = [np.ones_like(_xc, dtype=int) * 0 for _xc in xc]

        m = [_rho * (self.dx/2.)**3 for _rho in rho]
        rad_s = self.dx / 4.
        V = (self.dx/2.)**3
        cs = 0.0
        cube = [
            get_particle_array_rigid_body(
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
        kernel = CubicSpline(dim=3)
        # kernel = WendlandQuintic(dim=3)

        # simulation = Simulation()
        # cubes = {
        #     "cube_{}".format(i): RK2StepRigidBody(simulation=simulation)
        #     for i in range(12)
        # }
        integrator = EPECIntegrator(
            fluid=WCSPHStep(),
            tank=WCSPHStep(),
            # **cubes
        )

        # dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        print("DT: %s" % self.dt)
        tf = 1.2  # 5
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=self.dt,
            tf=tf,
            adaptive_timestep=False,  # False because otherwise not constant
            reorder_freq=0
        )
        solver.set_output_fname("salamander")  # Does not work
        return solver

    def create_equations(self):
        # simulation = Simulation()
        # cube_gravity = [  # Replace this!!
        #     Group(
        #         equations=[
        #             BodyForce(
        #                 dest='cube_{}'.format(i),
        #                 sources=None,
        #                 simulation=self.simulation,
        #                 gz=-9.81
        #             )
        #             for i in range(12)
        #         ],
        #         real=False
        #     )
        # ]
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
        # cube_collisions = [  # Replace this!!
        #     Group(
        #         equations=[
        #             RigidBodyCollision(
        #                 dest='cube_{}'.format(i),
        #                 sources=['tank', 'cube_{}'.format(i)],
        #                 kn=1e3
        #             )
        #             for i in range(12)
        #         ]
        #     )
        # ]
        cube_moment = [
            Group(
                equations=[
                    RigidBodyMoments(
                        dest='cube_{}'.format(i),
                        sources=None
                    )
                    for i in range(12)
                ]
            )
        ]
        # cube_motion = [  # Replace this!!
        #     Group(
        #         equations=[
        #             RigidBodyMotion(
        #                 dest='cube_{}'.format(i),
        #                 sources=None
        #             )
        #             for i in range(12)
        #         ]
        #     )
        # ]

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

        equations = (
            # cube_gravity
            continuity
            + tait
            + fluid_motion
            + cube_moment
            + bullet_forces
            + bullet_update
            + bullet_motion
            # + cube_collisions
            # + cube_moment
            # + cube_motion
        )
        return equations


def main():
    """Main"""
    # for n_particles in [10000, 50000, 100000]:  # 500000
    #     for density in [0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 2.0]:
    #         try:
    #             app = RigidFluidCoupling(
    #                 density=density,
    #                 n_fluid_particles=n_particles,
    #                 output_dir="benchmark_d{}_n{}".format(density, n_particles)
    #             )
    #             app.run()
    #         except Exception as err:
    #             print(str(err))
    # for n_particles in [10000, 50000, 100000, 500000]:  # , 50000, 100000
    for n_particles in [1000]:
        # for density in [500, 800, 900, 1000, 1100, 1200, 2000]:
        for density in [1000]:
            try:
                print("Density: {}".format(density))
                app = RigidFluidCoupling(
                    density=density,
                    n_fluid_particles=n_particles,
                    output_dir="benchmark_d{}_n{}".format(
                        density,
                        n_particles
                    )
                )
                app.run()
            except Exception as err:
                print("WATCH OUT THERE WAS AN ERROR!!!")
                traceback.print_exc(file=sys.stdout)
                print(str(err))


if __name__ == '__main__':
    main()
    # print(mesh_to_particles())
    # print(get_3d_block())
