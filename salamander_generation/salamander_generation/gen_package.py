""" Generate package """

import os
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader


class PackageGenerationTemplates(object):
    """ Diagram templates """

    def __init__(self):
        super(PackageGenerationTemplates, self).__init__()
        self.env = Environment(loader=FileSystemLoader(
            os.path.join(os.path.dirname(__file__), "templates")
        ))
        self.package = self.env.get_template('package.yaml')
        self.plugin = self.env.get_template('plugin.yaml')

    def render(self, **kwargs):
        """ Render Model """
        package = self.package.render(**kwargs)
        path = os.path.join(os.path.dirname(__file__), "config", "package.yaml")
        if kwargs.pop("verbose", False):
            print("Package: {}".format(package))
        with open(path, "w+") as fileobject:
            fileobject.write(package)
            print("Generation of {} complete".format(path))


def generate_package(name, model_name, base_model, gait):
    """ Generate package """
    package = PackageGenerationTemplates()
    plugins = [
        package.plugin.render(
            plugin="control",
            name="{}_controller".format(model_name),
            filename="libbiorob_salamander_control_plugin.so",
            config="control.yaml"
        )
    ] + (
        [
            package.plugin.render(
                plugin="viscous_swimming",
                name="viscous_swimming",
                filename="libbiorob_salamander_viscous_swimming_plugin.so",
                config="swim.yaml"
            )
        ]
        if gait == "swimming" else []
    ) + [
        package.plugin.render(
            plugin="log_kinematics",
            name="log_kinematics",
            filename="libbiorob_salamander_log_kinematics_plugin.so",
            config="log_kinematics.yaml",
            config_file="config_log_kinematics.yaml"
        )
    ]
    package.render(
        name=name,
        model_name=model_name,
        base_model=base_model,
        plugins="\n".join(plugins)
    )


def main():
    """ Main """
    name = "salamander_walking"
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model="biorob_salamander",
        gait="walking"
    )


if __name__ == '__main__':
    main()
