"""3D benchmark with rigid boy physics coupling

A cube with a specified density falls into a pool of water. This is based on the
three_cubes_in_vessel_3d.py example provided by PySPH.

"""
from __future__ import print_function
import sys
# import os
import numpy as np

from pysph.base.utils import (get_particle_array_wcsph,
                              get_particle_array_rigid_body)
# PySPH base and carray imports
# from pysph.base.kernels import CubicSpline
from pysph.base.kernels import WendlandQuintic

from pysph.solver.solver import Solver
from pysph.sph.integrator import EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

from pysph.sph.equation import Group
from pysph.sph.basic_equations import (XSPHCorrection, ContinuityEquation)
from pysph.sph.wc.basic import TaitEOSHGCorrection, MomentumEquation
from pysph.solver.application import Application
from pysph.sph.rigid_body import (
    BodyForce, RigidBodyCollision, RigidBodyMoments,
    RigidBodyMotion, AkinciRigidFluidCoupling, RK2StepRigidBody)


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


class RigidFluidCoupling(Application):
    """Rigid body to fluid coupling"""

    def __init__(self, density=500, n_fluid_particles=8000, output_dir=None):
        self.density = density
        self.n_fluid_particles = n_fluid_particles
        super(RigidFluidCoupling, self).__init__()
        if output_dir:
            self.args.append("--directory")
            self.args.append(output_dir)

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
        link_positions = [  # From SDF
            [0, 0, 0],
            [0.200000003, 0, 0.0069946074],
            [0.2700000107, 0, 0.010382493],
            [0.3400000036, 0, 0.0106022889],
            [0.4099999964, 0, 0.010412137],
            [0.4799999893, 0, 0.0086611426],
            [0.5500000119, 0, 0.0043904358],
            [0.6200000048, 0, 0.0006898994],
            [0.6899999976, 0, 8.0787e-06],
            [0.7599999905, 0, -4.89001e-05],
            [0.8299999833, 0, 0.0001386079],
            [0.8999999762, 0, 0.0003494423]
        ]
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
        # kernel = CubicSpline(dim=3)
        kernel = WendlandQuintic(dim=3)

        cubes = {
            "cube_{}".format(i): RK2StepRigidBody()
            for i in range(12)
        }
        integrator = EPECIntegrator(
            fluid=WCSPHStep(),
            tank=WCSPHStep(),
            **cubes
        )

        # dt = 0.125 * self.dx * self.hdx / (self.co * 1.1) / 2.
        dt = 1e-3
        print("DT: %s" % dt)
        tf = 1  # 5
        solver = Solver(
            kernel=kernel,
            dim=3,
            integrator=integrator,
            dt=dt,
            tf=tf,
            adaptive_timestep=False,  # False
        )
        return solver

    def create_equations(self):
        cube_gravity = [
            Group(
                equations=[
                    BodyForce(
                        dest='cube_{}'.format(i),
                        sources=None,
                        gz=-9.81),
                ],
                real=False
            )
            for i in range(12)
        ]
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
        cube_moment = [
            Group(
                equations=[
                    RigidBodyMoments(
                        dest='cube_{}'.format(i),
                        sources=None
                    )
                ]
            )
            for i in range(12)
        ]
        cube_motion = [
            Group(
                equations=[
                    RigidBodyMotion(
                        dest='cube_{}'.format(i),
                        sources=None
                    )
                ]
            )
            for i in range(12)
        ]
        cube_collisions = [
            Group(
                equations=[
                    RigidBodyCollision(
                        dest='cube_{}'.format(i),
                        sources=['tank', 'cube_{}'.format(i)],
                        kn=1e5
                    )
                ]
            )
            for i in range(12)
        ]
        equations = (
            cube_gravity
            + continuity
            + tait
            + fluid_motion
            + cube_moment
            + cube_motion
            + cube_collisions
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
