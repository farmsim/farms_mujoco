""" OpenMPI communication """

from mpi4py import MPI

class MPIsettings(dict):
    """ MPI settings """

    def __init__(self):
        super(MPIsettings, self).__init__()
        self["comm"] = MPI.COMM_WORLD
        self["size"] = self.comm.Get_size()
        self["rank"] = self.comm.Get_rank()

    @property
    def comm(self):
        """ Comm """
        return self["comm"]

    @property
    def size(self):
        """ Size """
        return self["size"]

    @property
    def rank(self):
        """ Rank """
        return self["rank"]


