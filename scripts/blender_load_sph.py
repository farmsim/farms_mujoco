import time

import bpy
import bmesh

import numpy as np


def particles_as_objects():
    """Main"""

    tic = time.time()

    # verts = [(1, 1, 1), (0, 0, 0)]  # 2 verts made with XYZ coords
    dist = 10
    n_parts = 10
    scale = 1
    verts = [
        (i, j, k)
        for i in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for j in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for k in np.linspace(-dist, dist, n_parts)
    ]

    # for vert in verts:
    #     bpy.ops.mesh.primitive_ico_sphere_add(
    #         subdivisions=1,
    #         radius=1,
    #         calc_uvs=False,
    #         enter_editmode=False,
    #         align='WORLD',
    #         location=vert,
    #         rotation=(0, 0, 0)
    #     )
    #     # bpy.ops.surface.primitive_nurbs_surface_sphere_add(
    #     #     radius=1,
    #     #     enter_editmode=False,
    #     #     align='WORLD',
    #     #     location=vert,
    #     #     rotation=(0, 0, 0)
    #     # )

    # scene = bpy.context.scene
    # scene.objects.link(obj)  # put the object into the scene (link)
    # scene.objects.active = obj  # set as the active object in the scene
    # obj.select = True  # select object

    # Create collection
    collection_name = "Particles"
    if collection_name not in bpy.data.collections:
        bpy.data.collections.new(collection_name)
    collection = bpy.data.collections[collection_name]
    if collection_name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(collection)

    # # Create mesh
    # mesh = bpy.data.meshes.new("mesh")  # add a new mesh
    # obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh
    # # Add mesh to collection
    # collection.objects.link(obj)
    # # mesh = bpy.context.object.data
    # mesh = obj.data
    # bm = bmesh.new()
    # for v in verts:
    #     bm.verts.new(v)  # add a new vert
    # # make the bmesh the object's mesh
    # bm.to_mesh(mesh)
    # bm.free()  # always do this when finished

    # # Add sphere
    # # bpy.ops.surface.primitive_nurbs_surface_sphere_add()
    # bpy.ops.mesh.primitive_ico_sphere_add(
    #     subdivisions=1,
    #     radius=1,
    #     calc_uvs=True,
    #     enter_editmode=False,
    #     align='WORLD',
    #     location=(0, 0, 0),
    #     rotation=(0, 0, 0)
    # )
    # sphere = bpy.context.object
    # sphere.name = "sphere"
    # sphere.parent = obj

    # # Dupliverts
    # obj.instance_type = 'VERTS'

    # # Particles
    # # bpy.ops.mesh.select_all(action='DESELECT')
    # particles_mod = obj.modifiers.new(name="ParticleSystem", type='PARTICLE_SYSTEM')
    # particles_sys = particles_mod.particle_system
    # # obj.modifier_add(type='PARTICLE_SYSTEM')
    # # bpy.context.space_data.context = 'PARTICLES'
    # bpy.data.particles["ParticleSettings"].emit_from = 'VERT'
    # bpy.data.particles["ParticleSettings"].physics_type = 'NO'

    # Create materials
    materials = [None for _ in range(10)]
    for i, _ in enumerate(materials):
        color = np.random.ranf(4)
        color_name = "color{}".format(i)
        materials[i] = bpy.data.materials.new(name=color_name)
        materials[i].diffuse_color = color

    # Create original element
    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=0,
        radius=1,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        rotation=(0, 0, 0)
    )
    # bpy.ops.surface.primitive_nurbs_surface_sphere_add(
    #     radius=1,
    #     enter_editmode=False,
    #     align='WORLD',
    #     location=(0, 0, 0),
    #     rotation=(0, 0, 0)
    # )
    sphere = bpy.context.object
    sphere.name = "original_sphere"
    collection.objects.link(sphere)
    spheres = [None for _ in verts]

    # # Add materials to base element
    # for i, material in enumerate(materials):
    #     print("Setting material {}".format(i))
    #     sphere.data.materials.append(material)
    # # for mat_slot in sphere.material_slots:
    # #     mat = mat_slot.material
    # #     mat_slot.link = 'OBJECT'
    # #     mat_slot.material = mat
    # for _material in sphere.material_slots:
    #     _material.link = 'OBJECT'
    # # sphere.active_material = materials[0]

    # Copy sphere
    for i, vert in enumerate(verts):
        print("Copying sphere {}".format(i))
        # spheres[i] = sphere.copy()
        spheres[i] = bpy.data.objects.new("sphere_{}".format(i), sphere.data)

    # Link to collection
    for i, vert in enumerate(verts):
        print("Linking sphere to collection {}".format(i))
        collection.objects.link(spheres[i])

    # Set positions
    for i, vert in enumerate(verts):
        spheres[i].location[0] = vert[0]
        spheres[i].location[1] = vert[1]
        spheres[i].location[2] = vert[2]

    # Set materials
    for i, vert in enumerate(verts):
        print("Setting material {}".format(i))
        material = materials[np.random.randint(low=0, high=9)]
        for _material in spheres[i].material_slots:
            _material.link = 'OBJECT'
        spheres[i].active_material = material

    # # Keyframes
    # for i, vert in enumerate(verts):
    #     print("Setting location and keyframes {}".format(i))
    #     for frame in range(10):
    #         # Set location
    #         spheres[i].location[0] = vert[0] + 10*np.sin(2*np.pi*0.1*frame)
    #         spheres[i].location[1] = vert[1] + 10*np.cos(2*np.pi*0.1*frame)
    #         spheres[i].location[2] = vert[2] + 10*np.sin(2*np.pi*0.1*frame)
    #         for dim in range(3):
    #             spheres[i].keyframe_insert(
    #                 data_path='location',
    #                 index=dim,
    #                 frame=frame
    #             )
    #         # Set material
    #         material = materials[np.random.randint(low=0, high=9)]
    #         for dim in range(3):
    #             spheres[i].keyframe_insert(
    #                 data_path='location',
    #                 index=dim,
    #                 frame=frame
    #             )

    toc = time.time()

    print("Total time for {} particles: {} [s]".format(len(verts), toc-tic))


def particles_as_particles():
    """Main"""

    tic = time.time()

    dist = 10
    n_parts = 10
    scale = 3
    verts = [
        (i, j, k)
        for i in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for j in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for k in np.linspace(-dist, dist, n_parts)
    ]

    # Create collection
    collection_name = "Particles"
    if collection_name not in bpy.data.collections:
        bpy.data.collections.new(collection_name)
    collection = bpy.data.collections[collection_name]
    if collection_name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(collection)

    bpy.ops.mesh.primitive_ico_sphere_add(
        subdivisions=0,
        radius=1,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        rotation=(0, 0, 0)
    )
    sphere = bpy.context.object
    sphere.name = "original_sphere"
    collection.objects.link(sphere)

    # Create original element
    bpy.ops.mesh.primitive_cube_add(
        size=2,
        calc_uvs=True,
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        rotation=(0, 0, 0)
    )
    domain = bpy.context.object
    domain.name = "domain"
    collection.objects.link(domain)

    # Create materials
    materials = [None for _ in range(10)]
    for i, _ in enumerate(materials):
        color = np.random.ranf(4)
        color_name = "color{}".format(i)
        materials[i] = bpy.data.materials.new(name=color_name)
        materials[i].diffuse_color = color

    # Set materials
    for material in materials:
        domain.data.materials.append(material)

    # Particles
    particles_mod = domain.modifiers.new(
        name="ParticleSystem",
        type='PARTICLE_SYSTEM'
    )
    # particles_sys = particles_mod.particle_system
    particles_sys = bpy.data.particles["ParticleSystem"]
    particles_sys.count = len(verts)
    particles_sys.particle_size = 1
    particles_sys.frame_start = 1
    particles_sys.frame_end = 1
    particles_sys.lifetime = 1000
    particles_sys.emit_from = 'VERT'
    particles_sys.physics_type = 'NO'
    particles_sys.render_type = 'OBJECT'
    particles_sys.instance_object = sphere
    # particles = particles_sys.particles


    # Dependancy graph
    degp = bpy.context.evaluated_depsgraph_get()

    # Evaluate the depsgraph (Important step)
    particle_systems = domain.evaluated_get(degp).particle_systems

    # All particles of first particle-system which has index "0"
    particles = particle_systems[0].particles

    # Total Particles
    totalParticles = len(particles)

    particles.foreach_set("location", np.array(verts).reshape([3*len(verts)]))

    # length of 1D array or list = 3*totalParticles, "3" due to XYZ in vector/location.
    # If the length is wrong then it will give you an error "internal error setting the array"
    flatList = [0]*(3*totalParticles)

    # To get the loaction of all particles
    particles.foreach_get("location", flatList)

    print(flatList)

    # ps = bpy.context.object.particle_systems[0]
    # particles = ps.particles

    # # Set positions
    # for i, vert in enumerate(verts):
    #     spheres[i].location[0] = vert[0]
    #     spheres[i].location[1] = vert[1]
    #     spheres[i].location[2] = vert[2]

    # # Set materials
    # for i, vert in enumerate(verts):
    #     print("Setting material {}".format(i))
    #     material = materials[np.random.randint(low=0, high=9)]
    #     for _material in spheres[i].material_slots:
    #         _material.link = 'OBJECT'
    #     spheres[i].active_material = material

    toc = time.time()

    print("Total time for {} particles: {} [s]".format(len(verts), toc-tic))


def particles_as_dupliverts():
    """Main"""

    tic = time.time()

    # verts = [(1, 1, 1), (0, 0, 0)]  # 2 verts made with XYZ coords
    dist = 10
    n_parts = 10
    scale = 10
    verts = np.array([
        (i, j, k)
        for i in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for j in np.linspace(-scale*dist, scale*dist, scale*n_parts)
        for k in np.linspace(-dist, dist, n_parts)
    ])

    # Create collection
    collection_name = "Particles"
    if collection_name not in bpy.data.collections:
        bpy.data.collections.new(collection_name)
    collection = bpy.data.collections[collection_name]
    if collection_name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(collection)

    n_particles = len(verts)
    n_sets = 64
    indices = np.random.permutation(n_particles)
    materials = [None for _ in range(n_sets)]
    for particle_set in range(n_sets):

        # Create mesh
        mesh = bpy.data.meshes.new("mesh")  # add a new mesh
        obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh
        # Add mesh to collection
        collection.objects.link(obj)
        # mesh = bpy.context.object.data
        mesh = obj.data
        bm = bmesh.new()
        for v in verts[indices][
                (particle_set*n_particles)//n_sets
                :((particle_set+1)*n_particles)//n_sets
        ]:
            bm.verts.new(v)  # add a new vert
        # make the bmesh the object's mesh
        bm.to_mesh(mesh)
        bm.free()  # always do this when finished

        # Add sphere
        # bpy.ops.surface.primitive_nurbs_surface_sphere_add()
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=0,
            radius=1,
            calc_uvs=True,
            enter_editmode=False,
            align='WORLD',
            location=(0, 0, 0),
            rotation=(0, 0, 0)
        )
        sphere = bpy.context.object
        sphere.name = "sphere"
        sphere.parent = obj

        # Dupliverts
        obj.instance_type = 'VERTS'

        # Create materials
        color = np.random.ranf(4)
        color_name = "color{}".format(particle_set)
        materials[particle_set] = bpy.data.materials.new(name=color_name)
        materials[particle_set].diffuse_color = color
        sphere.data.materials.append(materials[particle_set])
        sphere.active_material = materials[particle_set]

    toc = time.time()

    print("Total time for {} particles: {} [s]".format(len(verts), toc-tic))


def profile():
    """Profile with cProfile"""
    import cProfile
    import pstats
    cProfile.run("import bpy; main()", "simulation.profile")
    pstat = pstats.Stats("simulation.profile")
    pstat.sort_stats('time').print_stats(30)
    pstat.sort_stats('cumtime').print_stats(30)


if __name__ == '__main__':
    particles_as_dupliverts()
