""" Generate package """

from .model import (
    generate_model_options,
    ModelGenerationTemplates
    # Sensors
)
from .gen_controller import ControlParameters


def generate_walking(name, control_parameters):
    """ Generate walking salamander """
    package = generate_model_options(
        name=name,
        base_model="biorob_salamander",
        gait="walking",
        control_parameters=control_parameters
    )
    templates = ModelGenerationTemplates()
    packager = templates.render(package)
    packager.generate()


def generate_swimming(name, control_parameters):
    """ Generate walking salamander """
    package = generate_model_options(
        name=name,
        base_model="biorob_salamander_slip",
        gait="swimming",
        control_parameters=control_parameters
    )
    templates = ModelGenerationTemplates()
    packager = templates.render(package)
    packager.generate()


def generate_swimming_legless(name, control_parameters):
    """ Generate walking salamander """
    package = generate_model_options(
        name=name,
        base_model="biorob_salamander_slip_no_legs",
        gait="swimming",
        control_parameters=control_parameters
    )
    templates = ModelGenerationTemplates()
    packager = templates.render(package)
    packager.generate()


def generate_all():
    """ Test entity generation """
    name = "salamander_walking"
    generate_walking(
        name,
        ControlParameters(gait="walking", frequency=1)
    )
    generate_walking(
        name+"_slow",
        ControlParameters(gait="walking", frequency=0.5)
    )
    generate_walking(
        name+"_fast",
        ControlParameters(gait="walking", frequency=2)
    )
    name = "salamander_swimming"
    generate_swimming(
        name,
        ControlParameters(gait="swimming", frequency=2)
    )
    generate_swimming_legless(
        name+"_legless",
        ControlParameters(
            gait="swimming",
            frequency=1.5,
            log_frequency=1000,
            legs=False
        )
    )


if __name__ == '__main__':
    generate_all()
