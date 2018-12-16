""" Generate all models """

from .gen_package import generate_package
from .gen_plugins import generate_plugins
from .gen_model import generate_model


def generate_walking(name, base_model, frequency):
    """ Generate walking salamander """
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="walking"
    )
    generate_plugins(gait="walking", frequency=frequency)
    generate_model()


def generate_swimming(name, base_model, frequency):
    """ Generate walking salamander """
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="walking"
    )
    generate_plugins(gait="walking", frequency=frequency)
    generate_model()


def generate_all():
    """ Main """
    # Walking
    name = "salamander_walking"
    base_model = "biorob_salamander"
    generate_walking(name, base_model, frequency=1)
    generate_walking(name+"_slow", base_model, frequency=0.5)
    generate_walking(name+"_fast", base_model, frequency=2)

    # Swimming
    name = "salamander_swimming"
    base_model = "biorob_salamander_slip"
    generate_walking(name, base_model=base_model, frequency=2) # Fast


if __name__ == '__main__':
    generate_all()
