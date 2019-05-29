"""Network"""

class NetworkParameters:
    """Network parameter"""

    def __init__(
            self,
            oscillators,
            connectivity,
            joints,
            contacts,
            contacts_connectivity
    ):
        super(NetworkParameters, self).__init__()
        self.function = [
            oscillators,
            connectivity,
            joints,
            contacts,
            contacts_connectivity
        ]

    @property
    def oscillators(self):
        """Oscillators parameters"""
        return self.function[0]

    @property
    def connectivity(self):
        """Connectivity parameters"""
        return self.function[1]

    @property
    def joints(self):
        """Joints parameters"""
        return self.function[2]

    @property
    def contacts(self):
        """Contacts parameters"""
        return self.function[3]

    @property
    def contacts_connectivity(self):
        """Contacts parameters"""
        return self.function[4]

    def to_ode_parameters(self):
        """Convert 2 arrays"""
        return (
            [parameter.array for parameter in self.function]
            + [self.oscillators.shape()[1]]
            + [self.connectivity.shape()[0]]
            + [self.joints.shape()[1]]
            + [self.contacts.shape()[1]]
            + [self.contacts_connectivity.shape()[0]]
            + [0]
        )
