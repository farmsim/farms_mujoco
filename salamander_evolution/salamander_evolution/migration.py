"""Migration"""

from abc import abstractmethod

import numpy as np


class Migration:
    """Migration"""

    def __init__(self, topology):  # , check=True
        super(Migration, self).__init__()
        self._topology = topology
        # if check:
        #     self.check()

    @abstractmethod
    def apply(self, pops, gen):
        """Apply migration between islands"""

    @property
    def topology(self):
        """Topology"""
        return np.copy(self._topology)

    # def check(self):
    #     """Check topology"""


class RingMigration(Migration):
    """RingMigration"""

    def __init__(self, n_islands, p_migrate_backward, p_migrate_forward):
        self.p_migrate_backward = p_migrate_backward
        self.p_migrate_forward = p_migrate_forward
        topology = np.zeros([n_islands, n_islands])
        # Forward
        for i in range(n_islands-1):
            topology[i][i+1] = p_migrate_forward
        topology[-1][0] = p_migrate_forward
        # Backward
        for i in range(n_islands-1):
            topology[i+1][i] = p_migrate_backward
        topology[0][-1] = p_migrate_backward
        super(RingMigration, self).__init__(topology)

    def apply(self, pops, gen):
        for i_isl_origin, _ in enumerate(pops):
            for i_isl_destination, _ in enumerate(pops[:-1]):
                if self.topology[i_isl_origin][i_isl_destination] > 0:
                    pops[i_isl_destination][gen].set_xf(
                        pops[i_isl_destination][gen].worst_idx(),
                        pops[i_isl_origin][gen].champion_x,
                        pops[i_isl_origin][gen].champion_f
                    )
