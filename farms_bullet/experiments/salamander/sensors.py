"""GPS for salamander animat"""

from ...sensors.sensors import LinksStatesSensor


class SalamanderGPS(LinksStatesSensor):
    """Salamander GPS"""

    def __init__(self, array, animat_id, links, options):
        super(SalamanderGPS, self).__init__(
            array=array,
            animat_id=animat_id,
            links=links
        )
        self.options = options

    def update(self, iteration):
        """Update sensor"""
        if self.options.collect_gps:
            self.collect(iteration, self.links)
        if self.options.control.drives.forward > 3:
            self.collect(iteration, self.links[:12])
