""" Generate salamander model """

import os
from shutil import copyfile
# from collections import Iterable
from collections import OrderedDict

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from .yaml_utils import ordered_load


class PluginParameters:
    """Plugin parameters"""

    def __init__(self, name, filename, config_file):
        super(PluginParameters, self).__init__()
        self.name = name
        self.filename = filename
        self.parameters = get_config(filename=filename)
        self.config_file = config_file


class ModelGenerator:
    """ Gazebo model configuration """

    def __init__(self, config):
        super(ModelGenerator, self).__init__()
        self.config_package = get_config(filename=config)
        plugins = self.config_package["model"]["plugins"]
        plugins_parameters = [
            PluginParameters(
                name=plugin,
                filename="config/{}".format(
                    plugins[plugin]["config"]
                ),
                config_file=plugins[plugin]["config_file"]
            )
            for plugin in plugins
        ]
        self.config_plugins = plugins_parameters

    def generate(self):
        """Generate SDF"""
        templates = ModelGenerationTemplates()
        templates.render(
            self.config_package,
            self.config_plugins
        )


def create_directory(folder):
    """ Create directory if it does not exist """
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        print("{} does not exist, creating directory".format(directory))
        os.makedirs(directory)


class ModelGenerationTemplates:
    """ Diagram templates """

    def __init__(self):
        super(ModelGenerationTemplates, self).__init__()
        self.env = Environment(loader=FileSystemLoader(
            os.path.join(os.path.dirname(__file__), "templates")
        ))
        self.sdf = self.env.get_template('salamander.sdf')
        self.parameter = self.env.get_template('parameters.xml')
        self.model = self.env.get_template('model.config')
        self.plugin = self.env.get_template('plugin.xml')

    def render(self, config_package, plugins):
        """ Render Model """
        model_name = config_package["model"]["model_name"]
        folder_path = (
            os.path.expanduser("~")
            + "/{}{}/".format(
                config_package["gazebo"]["models_path"],
                model_name
            )
        )
        create_directory(folder_path)
        # Get environment and templates
        filename_sdf = config_package["model"]["sdf_name"]
        filename_model = config_package["model"]["config_name"]
        # Generate sdf
        # parameters = self.get_parameters(config_control)
        _plugins = "\n".join([
            self.plugin.render(
                plugin_filename=(
                    config_package["model"]["plugins"][plugin.name]["filename"]
                ),
                plugin_name=(
                    config_package["model"]["plugins"][plugin.name]["name"]
                ),
                parameters="\n".join(self.get_parameters(plugin.parameters))
            )
            for plugin in plugins
        ])
        sdf = self.sdf.render(
            name=config_package["model"]["name"],
            plugins=_plugins,
            base_model=config_package["model"]["base_model"]
        )
        with open(folder_path+filename_sdf, "w+") as fileobject:
            fileobject.write(sdf)
            print("Generation of {} sdf complete".format(filename_sdf))
        # Generate model config
        model = self.model.render(
            model_name=model_name,
            filename_sdf=filename_sdf,
            version=config_package["model"]["version"],
            author=config_package["creator"]["author"],
            email=config_package["creator"]["email"]
        )
        with open(folder_path+filename_model, "w+") as fileobject:
            fileobject.write(model)
            print("Generation of {} model complete".format(filename_model))
        for plugin in plugins:
            if plugin.config_file:
                dest = folder_path+plugin.filename
                src = os.path.join(
                    os.path.dirname(__file__),
                    "config",
                    plugin.config_file
                )
                print("Copying config file {}".format(src))
                create_directory(dest)
                copyfile(src, dest)

    def get_parameters(self, config):
        """ Get parameters """
        parameters = []
        for _name in config:
            self.get_parameter(parameters, _name, config[_name])
        return parameters

    def get_parameter(
            self, parameters, name, value, prefix="", sep="_"
    ):
        """ Get parameter """
        if isinstance(value, (dict, OrderedDict)):
            for _name in value:
                self.get_parameter(
                    parameters,
                    _name,
                    value[_name],
                    prefix=prefix+name+sep,
                    sep=sep
                )
        else:
            parameters.append(
                self.parameter.render(
                    parameter_name=prefix+name,
                    parameter_value=value
                )
            )
        return parameters


def get_config(filename):
    """ Get configuration file """
    _filename = os.path.join(os.path.dirname(__file__), filename)
    with open(_filename, 'r') as config_file:
        # config = yaml.load(config_file)
        config = ordered_load(config_file)
    return config


def generate_model():
    """ Generate model """
    print("Generating model")
    model_gen = ModelGenerator(config="config/package.yaml")
    model_gen.generate()
