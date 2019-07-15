"""SPH viewer with Blender"""

import os
import argparse
import cv2
import h5py

import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse aguments"""
    parser = argparse.ArgumentParser(description='View swimmming results')
    parser.add_argument(
        '-d', '--directory',
        dest='directory',
        action='store',
        default="",
        help='Directory with results'
    )
    parser.add_argument(
        '-m', '--movie',
        dest='movie',
        action='store',
        default="movie",
        help='Movie filename'
    )
    return parser.parse_args()


def get_files_from_extension(directory, extension):
    """Get hdf5 files"""
    print("Loading files from {}".format(directory))
    files = [
        filename
        for filename in os.listdir(directory)
        if extension in filename
    ]
    indices = [
        int("".join(filter(str.isdigit, filename.replace(extension, ""))))
        for filename in files
    ]
    files = [filename for _, filename in sorted(zip(indices, files))]
    print("Found files:\n{}".format(files))
    return files


def open_hdf5_file(filename):
    """Open hdf5 file"""
    print("Opening file {}".format(filename))
    return h5py.File(filename, 'r')


def view_sph(directory):
    """Load"""
    files = get_files_from_extension(directory, extension=".hdf5")
    n_body = 12
    n_files = len(files)
    forces = [None for _ in range(n_body)]
    for file_i, filename in enumerate(files):
        data = open_hdf5_file("{}/{}".format(directory, filename))
        print("Data keys: {}".format(data.keys()))
        particles = data["particles"]
        print("Particles keys: {}".format(particles.keys()))
        # Fluid
        fluid = particles["fluid"]
        print("Fluid keys: {}".format(fluid.keys()))
        fluid_arrays = fluid["arrays"]
        print("Fluid keys: {}".format(fluid_arrays.keys()))
        fluid_x = fluid_arrays["x"]
        print("Fluid x: {}".format(fluid_x))
        print("Fluid x[0]: {}".format(fluid_x[0]))
        for body in range(n_body):
            # Cube
            cube = particles["cube_{}".format(body)]
            print("Cube keys: {}".format(cube.keys()))
            cube_arrays = cube["arrays"]
            print("Cube keys: {}".format(cube_arrays.keys()))
            cube_fx = cube_arrays["fx"]
            print("Cube fx: {}".format(cube_fx))
            if not file_i:
                forces[body] = np.zeros(
                    [n_files, 3]
                    +list(np.shape(cube_arrays["fx"]))
                )
            forces[body][file_i, 0] = cube_arrays["fx"]
            forces[body][file_i, 1] = cube_arrays["fy"]
            forces[body][file_i, 2] = cube_arrays["fz"]
    # forces = [np.array(_forces) for _forces in forces]
    # print("Type forces: {}".format(forces.dtype))
    # print("Shape forces: {}".format(np.shape(forces)))
    # plt.plot(np.sum(np.linalg.norm(forces, axis=-1), axis=-1))
    for dim in range(3):
        plt.figure("Forces along {} axis".format(["X", "Y", "Z"][dim]))
        forces_total = np.zeros(n_files)
        for i, _forces in enumerate(forces):
            body_force = np.sum(_forces[:, dim, :], axis=-1)
            plt.plot(body_force, label="Body{}".format(i))
            forces_total += body_force
        plt.plot(forces_total, label="Total", linewidth=3)
        plt.legend()
    plt.show()


def view_video(directory, video_name='video'):
    """View video"""
    images = get_files_from_extension(directory, extension=".png")
    frame = cv2.imread(os.path.join(directory, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video = cv2.VideoWriter(
        filename=video_name+".mp4",
        fourcc=fourcc,
        fps=100/3,
        frameSize=(width,height)
    )
    for image in images:
        video.write(cv2.imread(os.path.join(directory, image)))
    cv2.destroyAllWindows()
    video.release()


def main():
    """Main"""
    clargs = parse_arguments()
    directory = clargs.directory
    view_sph(directory)
    # view_video(directory, video_name=clargs.movie)


if __name__ == "__main__":
    main()
