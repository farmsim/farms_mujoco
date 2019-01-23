"""Description"""

from collections import OrderedDict

import oyaml


class DescriptionElement(OrderedDict):
    """DesccriptionElement"""

    def __init__(self, **kwargs):
        super(DescriptionElement, self).__init__(**kwargs)

    def __str__(self):
        """Dump to YAML"""
        return self.dump_yaml()

    def dump(self):
        """Dump to OrderedDict"""
        return {
            self.__class__.__name__:
            OrderedDict([
                (item, [element.dump() for element in self[item]])
                if isinstance(self[item], list)
                else (item, self[item])
                for item in self
                if not isinstance(self[item], list) or len(self[item])
            ])
        }

    def dump_yaml(self):
        """Dump to YAML"""
        return oyaml.dump(self.dump(), default_flow_style=False)
