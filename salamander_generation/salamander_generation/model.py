"""Model"""

import os
from collections import OrderedDict

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader

from.yaml_utils import ordered_dump


def create_directory(folder):
    """ Create directory if it does not exist """
    directory = os.path.dirname(folder)
    if not os.path.exists(directory):
        print("{} does not exist, creating directory".format(directory))
        os.makedirs(directory)


def generate_plugins(plugins, folder_path):
    """ Generate plugins configs """
    for plugin in plugins:
        if plugin.filename:
            dest = "{}config/{}".format(
                folder_path,
                plugin.filename
            )
            if "filename" in plugin.config:
                plugin.config["filename"] = (
                    folder_path
                    + plugin.config["filename"]
                )
                create_directory(plugin.config["filename"])
            elif "logging" in plugin.config:
                if "filename" in plugin.config["logging"]:
                    plugin.config["logging"]["filename"] = (
                        folder_path
                        + plugin.config["logging"]["filename"]
                    )
                    create_directory(plugin.config["logging"]["filename"])
            create_directory(dest)
            with open(dest, "w+") as plugin_config:
                plugin_config.write(ordered_dump(plugin.config))


class JinjaGeneration:
    """JinjaGeneration"""

    def __init__(self):
        super(JinjaGeneration, self).__init__()
        self._path = os.path.join(os.path.dirname(__file__), "templates")
        self._env = Environment(loader=FileSystemLoader(self._path))

    @property
    def path(self):
        """Jinja templates path"""
        return self._path

    @property
    def env(self):
        """Jinja environment"""
        return self._env


class ModelFile:
    """Model file"""

    def __init__(self, filename, text):
        super(ModelFile, self).__init__()
        self._filename = filename
        self._text = text

    @property
    def filename(self):
        """Filename"""
        return self._filename

    @property
    def text(self):
        """Path"""
        return self._text


class ModelGenerationTemplates(JinjaGeneration):
    """Diagram templates"""

    def __init__(self):
        super(ModelGenerationTemplates, self).__init__()
        self.sdf = self.env.get_template('salamander.sdf')
        self.parameter = self.env.get_template('parameters.xml')
        self.model = self.env.get_template('model.config')
        # self.sensor = self.env.get_template('sensor.xml')
        self.plugin = self.env.get_template('plugin.xml')
        self.world = self.env.get_template('world.world')

    def render(self, config_package, sdf=None):
        """Render Model"""
        model_name = config_package.model.name
        folder_path = "/{}{}/".format(
            config_package.path,
            model_name
        )
        home = os.path.expanduser("~")
        create_directory(home+folder_path)
        # Get environment and templates
        filename_sdf = "model.sdf"
        filename_config = "model.config"
        filename_world = "world.world"
        # parameters = self.get_parameters(config_control)
        _plugins = "\n".join([
            self.plugin.render(
                plugin_filename=plugin.library,
                plugin_name=plugin.name,
                parameters="<config>{}config/{}</config>".format(
                    folder_path,
                    plugin.filename
                )
            )
            for plugin in config_package.model.plugins
        ]) if config_package.model.plugins else ""
        # _text = (
        #     '<joint name="sensor_joint" type="fixed">'
        #     + '\n  <parent>biorob_salamander::link_body_0</parent>'
        #     + '\n  <child>sensors</child>'
        #     + '\n</joint>'
        #     + '\n\n<link name="sensors">'
        #     + '\n{}'
        #     + '\n</link>'
        # )
        # _sensors = _text.format(
        #     "\n".join([
        #         self.sensor.render(
        #             sensor_name=sensor.name,
        #             sensor_type=sensor.type,
        #             always_on=sensor.always_on,
        #             update_rate=sensor.update_rate,
        #             visualize=sensor.visualize,
        #             topic=sensor.topic,
        #             collision=sensor.collision,
        #             contact_topic=sensor.contact_topic
        #         )
        #         for sensor in config_package.model.sensors
        #     ])) if config_package.model.sensors else ""
        # Generate sdf
        sdf = ModelFile(
            filename=filename_sdf,
            text=sdf if sdf is not None else self.generate_sdf(
                config_package,
                _plugins
                # _sensors
            )
        )
        # Generate model config
        model_config = ModelFile(
            filename=filename_config,
            text=self.generate_model_config(
                model_name=model_name,
                filename_sdf=filename_sdf,
                config_package=config_package
            )
        )
        # Generate world
        world = ModelFile(
            filename=filename_world,
            text=self.generate_world(config_package)
        )
        return ModelPackager(
            model_name=model_name,
            sdf=sdf,
            model_config=model_config,
            plugins=config_package.model.plugins,
            worlds=[world]
        )

    def generate_sdf(self, config_package, plugins):
        """Generate SDF"""
        return self.sdf.render(
            name=config_package.model.name,
            plugins=plugins,
            # sensors=sensors,
            base_model=config_package.model.base_model
        )

    def generate_model_config(self, config_package, **kwargs):
        """Generate model config"""
        model_name = kwargs.pop("model_name", "salamander_default_name")
        filename_sdf = kwargs.pop("filename_sdf", "model.sdf")
        return self.model.render(
            model_name=model_name,
            filename_sdf=filename_sdf,
            version=config_package.model.version,
            author=config_package.creator.name,
            email=config_package.creator.email
        )

    def generate_world(self, config_package):
        """Generate world"""
        return self.world.render(
            name="salamander",
            uri="model://{}".format(config_package.model.name),
            pose="0 0 0.1 0 0 0"
        )


class GenSDFinheritance(JinjaGeneration):
    """GenSDFinheritance"""

    def __init__(self, filename, config_package, path, **kwargs):
        super(GenSDFinheritance, self).__init__()
        self.sdf = self.env.get_template('salamander.sdf')
        self.filename = filename
        self.config_package = config_package
        self.plugins = kwargs.pop("plugins", None)
        # self.sensors = kwargs.pop("sensors", None)
        self.path = path

    def render(self):
        """Generate SDF"""
        sdf = self.sdf.render(
            name=self.config_package.model.name,
            plugins=self.plugins,
            # sensors=self.sensors,
            base_model=self.config_package.model.base_model
        )
        with open(self.path+self.filename, "w+") as fileobject:
            fileobject.write(sdf)
            print("  Generation of {} sdf complete".format(self.filename))


class GenSDFtext(JinjaGeneration):
    """GenSDFtext"""

    def __init__(self, sdf_text, filename, path):
        super(GenSDFtext, self).__init__()
        self.sdf = self.env.get_template('salamander_text.sdf')
        self.filename = filename
        self.path = path
        self.sdf_text = sdf_text

    @classmethod
    def from_model_name(cls, sdf_text, filename, model_name):
        """Generate sdf from model name (Placed in default gazebo location)"""
        path = ".gazebo/models/{}".format(model_name)
        return cls(sdf_text, filename, path)

    def render(self):
        """Generate SDF"""
        sdf = self.sdf.render(text=self.sdf_text)
        with open(self.path+self.filename, "w+") as fileobject:
            fileobject.write(sdf)
            print("  Generation of {} sdf complete".format(self.filename))


class ModelPackager:
    """ModelPackager"""

    def __init__(self, model_name, sdf, model_config, **kwargs):
        super(ModelPackager, self).__init__()
        self.model_name = model_name
        self.sdf = sdf
        self.model_config = model_config
        self.plugins = kwargs.pop("plugins", None)
        self.worlds = kwargs.pop("worlds", None)
        self.path = kwargs.pop("path", "{}/{}/{}/".format(
            os.path.expanduser("~"),
            ".gazebo/models",
            self.model_name
        ))

    def generate(self):
        """Generate package"""
        print("Generating {}".format(self.model_name))
        create_directory(self.path)
        self.generate_config()
        self.generate_sdf()
        self.generate_worlds()
        if self.plugins:
            generate_plugins(self.plugins, self.path)

    def generate_config(self):
        """Generate config"""
        print("  Generating {}".format(self.model_config.filename))
        with open(self.path+self.model_config.filename, "w+") as fileobject:
            fileobject.write(self.model_config.text)

    def generate_sdf(self):
        """Generate sdf"""
        print("  Generating {}".format(self.sdf.filename))
        with open(self.path+self.sdf.filename, "w+") as fileobject:
            fileobject.write(self.sdf.text)

    def generate_worlds(self):
        """Generate worlds"""
        for world in self.worlds:
            print("  Generating {}".format(world.filename))
            with open(self.path+world.filename, "w+") as fileobject:
                fileobject.write(world.text)


class Creator(OrderedDict):
    """ Creator """

    def __init__(self, name, email):
        super(Creator, self).__init__()
        self["name"] = name
        self["email"] = email

    @property
    def name(self):
        """ Name """
        return self["name"]

    @property
    def email(self):
        """ Email """
        return self["email"]


class Model(OrderedDict):
    """ Model """

    def __init__(self, name, base_model, plugins, version):
        super(Model, self).__init__()
        self["name"] = name
        self["base_model"] = base_model
        # self["sensors"] = sensors
        self["plugins"] = plugins
        self["version"] = version

    @property
    def name(self):
        """ Name """
        return self["name"]

    @property
    def base_model(self):
        """ Base model """
        return self["base_model"]

    # @property
    # def sensors(self):
    #     """ Sensors """
    #     return self["sensors"]

    @property
    def plugins(self):
        """ Plugins """
        return self["plugins"]

    @property
    def version(self):
        """ Version """
        return self["version"]


class Plugin(OrderedDict):
    """ Plugin """

    def __init__(self, name, library, config):
        super(Plugin, self).__init__()
        self["name"] = name
        self["library"] = library
        self["config"] = config
        self["filename"] = name+".yaml"

    @property
    def name(self):
        """ Name """
        return self["name"]

    @property
    def library(self):
        """ Library """
        return self["library"]

    @property
    def config(self):
        """ Config """
        return self["config"]

    @property
    def filename(self):
        """ Filename """
        return self["filename"]


class Plugins(list):
    """ Plugins """

    def __init__(self, gait, control_parameters):
        super(Plugins, self).__init__()
        self.gait = gait
        # Control
        self.append(
            Plugin(
                name="control",
                library="libbiorob_salamander_control2_plugin.so",
                config=control_parameters.data()
            )
        )
        if gait == "swimming":
            self.append(
                Plugin(
                    name="viscous_swimming",
                    library="libbiorob_salamander_viscous_swimming_plugin.so",
                    config={"swimming_parameter": True}
                )
            )
        self.append(
            Plugin(
                name="log_kinematics",
                library="libbiorob_salamander_log_kinematics_plugin.so",
                config={
                    "filename": "logs/links_kinematics.pbdat",
                    "links": OrderedDict(
                        [
                            (
                                "link_body_{}".format(i),
                                {"frequency": 100 if i == 0 else 10}
                            )
                            for i in range(12)
                        ]
                    ),
                    "joints": OrderedDict(
                        [
                            (
                                "joint_link_body_{}".format(i+1),
                                {"frequency": 100}
                            )
                            for i in range(11)
                        ]
                    )
                }
            )
        )


class Sensor(OrderedDict):
    """ Sensor """

    def __init__(self, name, sensor_type, **kwargs):
        self["name"] = name
        self["type"] = sensor_type
        self["always_on"] = kwargs.pop("always_on", True)
        self["update_rate"] = kwargs.pop("update_rate", 100)
        self["visualize"] = kwargs.pop("visualize", False)
        self["topic"] = kwargs.pop("topic", None)
        super(Sensor, self).__init__(**kwargs)

    @property
    def name(self):
        """ Name """
        return self["name"]

    @property
    def type(self):
        """ Type """
        return self["type"]

    @property
    def always_on(self):
        """ Always_On """
        return self["always_on"]

    @property
    def update_rate(self):
        """ Update_Rate """
        return self["update_rate"]

    @property
    def visualize(self):
        """ Visualize """
        return self["visualize"]

    @property
    def topic(self):
        """ Topic """
        return self["topic"]


class ContactSensor(Sensor):
    """Contact sensor """

    def __init__(self, collision, contact_topic, **kwargs):
        super(ContactSensor, self).__init__(**kwargs)
        self["collision"] = collision
        self["contact_topic"] = contact_topic

    @property
    def collision(self):
        """Collision"""
        return self["collision"]

    @property
    def contact_topic(self):
        """Contact topic"""
        return self["contact_topic"]


# class Sensors(list):
#     """ Sensors """

#     def __init__(self):
#         super(Sensors, self).__init__()
#         for name in [
#                 "leg_0_L_3",
#                 "leg_0_R_3",
#                 "leg_1_L_3",
#                 "leg_1_R_3"
#         ]:
#             self.append(
#                 ContactSensor(
#                     name="contact_{}".format(name),
#                     sensor_type="contact",
#                     always_on="true",
#                     update_rate=100,
#                     visualize="true",
#                     topic="/salamander/sensors/contact_{}".format(name),
#                     collision="biorob_salamander::collision_{}".format(name),
#                     contact_topic="/salamander/sensors/contacts/{}".format(name),
#                 )
#             )


class Package(OrderedDict):
    """ Model package """

    def __init__(self, creator, model, path=".gazebo/models/"):
        super(Package, self).__init__()
        self["creator"] = creator
        self["model"] = model
        self["path"] = path

    @property
    def creator(self):
        """ Creator """
        return self["creator"]

    @property
    def model(self):
        """ Model """
        return self["model"]

    @property
    def path(self):
        """ Path """
        return self["path"]

    def generate(self):
        """ Generate """


def generate_model_options(name, base_model, **kwargs):
    """ Generate package """
    gait = kwargs.pop("gait", None)
    control_parameters = kwargs.pop("control_parameters", None)
    creator = Creator(
        name="Jonathan Arreguit",
        email="jonathan.arreguitoneill@epfl.ch"
    )
    if bool(gait) != bool(control_parameters):
        raise Exception(
            "Gait must be defined with control_parameters"
            + " (gait: {} control_parameters: {}".format(
                gait,
                control_parameters
            )
        )
    elif gait and control_parameters:
        plugins = Plugins(gait, control_parameters)
    else:
        plugins = None
    # sensors = kwargs.pop("sensors", None)
    model = Model(
        name=name,
        base_model=base_model,
        plugins=plugins,
        # sensors=sensors,
        version="0.1"
    )
    package = Package(
        creator=creator,
        model=model,
        path=".gazebo/models/"
    )
    return package
