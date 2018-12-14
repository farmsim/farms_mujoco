""" Generate all models """

from .gen_package import generate_package
from .gen_plugins import generate_plugins
from .gen_model import generate_model


def generate_all():
    """ Main """
    base_model = "biorob_salamander"

    # Normal walking
    name = "salamander_walking"
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="walking"
    )
    generate_plugins(gait="walking")
    generate_model()

    # Slow walking
    name = "salamander_walking_slow"
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="walking"
    )
    generate_plugins(gait="walking", frequency=0.5)
    generate_model()

    # Fast walking
    name = "salamander_walking_fast"
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="walking"
    )
    generate_plugins(gait="walking", frequency=2)
    generate_model()

    # Swimming
    base_model = "biorob_salamander_slip"
    name = "salamander_swimming"
    generate_package(
        name=name,
        model_name="biorob_"+name,
        base_model=base_model,
        gait="swimming"
    )
    generate_plugins(gait="swimming", frequency=2)
    generate_model()


if __name__ == '__main__':
    generate_all()
