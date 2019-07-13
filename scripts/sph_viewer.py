"""SPH viewer with Blender"""

import os
import argparse
import cv2
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
    # view_sph(directory)
    view_video(directory, video_name=clargs.movie)


if __name__ == "__main__":
    main()
