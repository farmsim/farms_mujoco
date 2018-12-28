""" Install salamander framework """

import os
from subprocess import call


def create_build(dir_path):
    """ Create build """
    build = "build"
    file_path = "{}/{}".format(dir_path, build)
    directory = os.path.dirname(file_path)
    if build not in os.listdir(directory):
        print("Making directory {}".format(build))
        os.mkdir(build)
    else:
        print("Found directory {}".format(build))


def install_cpp(dir_path, n_processes=4, full=False):
    """ Build cpp codes """
    create_build(dir_path)
    os.chdir("{}/build".format(dir_path))
    if full:
        call("cmake ..", shell=True)
    call("make -j{}".format(int(n_processes)), shell=True)


def install_python(dir_path):
    """ Install python packages """
    for folder in os.listdir(dir_path):
        _folder = "{}/{}".format(dir_path, folder)
        if os.path.isdir(_folder) and "install.sh" in os.listdir(_folder):
            os.chdir(_folder)
            call("sh install.sh", shell=True)


def main():
    """ Main """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    install_cpp(dir_path)
    install_python(dir_path)


if __name__ == '__main__':
    main()
