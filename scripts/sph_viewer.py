"""SPH viewer with Blender"""

import os
import argparse
import h5py


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
    return parser.parse_args()


def get_hdf5_files(directory):
    """Get hdf5 files"""
    extension = ".hdf5"
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


def main():
    """Main"""
    clargs = parse_arguments()
    directory = clargs.directory
    files = get_hdf5_files(directory)
    for filename in [files[-1]]:
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
        # Cube
        cube = particles["cube_0"]
        print("Cube keys: {}".format(cube.keys()))
        cube_arrays = cube["arrays"]
        print("Cube keys: {}".format(cube_arrays.keys()))
        cube_fx = cube_arrays["fx"]
        print("Cube fx: {}".format(cube_fx))
        print("Cube fx[0]: {}".format(cube_fx[0]))


if __name__ == "__main__":
    main()
