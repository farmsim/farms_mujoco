"""Script1"""


def main(
        viewer,
        particle_arrays,
        interpolator,
        scene,
        mlab
):
    """Main"""
    print(particle_arrays)
    # Fluid
    fluid = particle_arrays['fluid']
    fluid.plot.actor.property.opacity = 0.1
    fluid.plot.actor.property.point_size = 15
    fluid.scalar = "vmag"
    fluid.show_legend = True
    fluid.show_time = True
    # Tank
    tank = particle_arrays['tank']
    tank.visible = False
    # Camera
    # scene.camera.roll(150)
    # scene.camera.pitch(100)
    scene.camera.roll(0)
    scene.camera.pitch(0)
    scene.camera.yaw(0)
    scene.camera.position = [-3, 3, 3]
    # Interpreter
    from IPython import embed; embed()


main(
    viewer,
    particle_arrays,
    interpolator,
    scene,
    mlab
)
