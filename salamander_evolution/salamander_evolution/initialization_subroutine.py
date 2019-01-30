import yaml
import pygmo as pg
import numpy as np

def isl_topology(isl_topology):
    """Read the right topology in the dictionnary of the config file"""
    for key, value in isl_topology.items():
        if value == True:
            isl_topo = key
    return isl_topo


def yaml_loader(filepath):
    """Loads a yaml file """
    with open(filepath, "r") as file_descriptor:
        data = yaml.load(file_descriptor)
    return data


def yaml_dump(filepath, data):
    """ Dumps data to yaml file """
    with open(filepath, "w") as file_descriptor:
        yaml.dump(data, file_descriptor)


def get_udas_name(udas):
    name_algo = []
    udas = [pg.algorithm(uda) for uda in udas]
    for m in np.arange(0, len(udas)):
        name_algo.append(udas[m].get_name())
    return name_algo


def init_de(de_params):
    """configuration of the user defined de algorithm"""
    #de_params = algo_config['de']
    uda = pg.de(gen=1, F=de_params['Muta_fac'],
                  CR=de_params['CR_fac'],
                  variant=de_params['variant'],
                  ftol=float(de_params['ftol']),
                  xtol=float(de_params['xtol']))
    return uda

def init_sga(sga_params):
    """configuration of the user defined sga algorithm"""
    #sga_params = algo_config['sga']
    uda = pg.sga(gen=1, cr=sga_params['CR_fac'],
                   eta_c=sga_params['eta_c'],
                   m=sga_params['Muta_prob'],
                   param_m=sga_params['param_m'],
                   param_s=sga_params['param_s'],
                   crossover=sga_params['crossover'],
                   mutation=sga_params['mutation'],
                   selection=sga_params['selection'])
    return uda


def init_sade(sade_params):
    """configuration of the user defined sade algorithm"""
    #sade_params = algo_config['sade']
    uda = pg.sade(gen=1, variant=sade_params['variant'],
                    variant_adptv=sade_params['variant_adptv'],
                    ftol=float(sade_params['ftol']),
                    xtol=float(sade_params['xtol']),
                    memory=sade_params['memory'])
    return uda


def init_pso(pso_params):
    #pso_params = algo_config['pso']
    uda = pg.pso(gen=1, omega=pso_params['omega'],
                   eta1=pso_params['eta1'],
                   eta2=pso_params['eta2'],
                   max_vel=pso_params['max_vel'],
                   variant=pso_params['variant'],
                   neighb_type=pso_params['neighb_type'],
                   neighb_param=pso_params['neighb_param'],
                   memory=pso_params['memory'])
    return uda


def init_bee_colony(bee_colony_params):
    #bee_colony_params = algo_config['bee_colony']
    uda = pg.bee_colony(gen=1, limit=bee_colony_params['limit'])
    return uda


def init_de1220(de1220_params):
    #de1220_params = algo_config['de1220']
    uda = pg.de1220(gen=1,
                      allowed_variants=de1220_params['allowed_variants'],
                      variant_adptv=de1220_params['variant_adptv'],
                      ftol=float(de1220_params['ftol']),
                      xtol=float(de1220_params['xtol']),
                      memory=de1220_params['memory'])
    return uda


def init_cmaes(cmaes_params):
    """initiation of the algorithm cmaes with the config.yalm"""
    uda = pg.cmaes(gen=1,
                     cc=cmaes_params['cc'],
                     cs=cmaes_params['cs'],
                     c1=cmaes_params['c1'],
                     cmu=cmaes_params['cmu'],
                     sigma0=cmaes_params['sigma0'],
                     ftol=float(cmaes_params['ftol']),
                     xtol=float(cmaes_params['xtol']),
                     memory=cmaes_params['memory'],
                     force_bounds=cmaes_params['force_bounds'])
    return uda


def init_pso_gen(pso_gen_params):
    """configuration of the user defined algorithm pygmo.pso_gen() with the parameters in entry"""
    #pso_gen_params = algo_config['psogen']
    uda = pg.pso_gen(gen=1,
                       omega=pso_gen_params['omega'],
                       eta1=pso_gen_params['eta1'],
                       eta2=pso_gen_params['eta2'],
                       max_vel=pso_gen_params['max_vel'],
                       variant=pso_gen_params['variant'],
                       neighb_type=pso_gen_params['neighb_type'],
                       neighb_param=pso_gen_params['neighb_param'],
                       memory=pso_gen_params['memory'])
    return uda


def init_xnes(xnes_params):
    """configuration of the user defined algorithm pygmo.xnes() with the parameters in entry"""
    #xnes_params = algo_config['xnes']
    uda = pg.xnes(gen=1,
                    eta_mu=xnes_params['eta_mu'],
                    eta_sigma=xnes_params['eta_sigma'],
                    eta_b=xnes_params['eta_b'],
                    sigma0=xnes_params['sigma0'],
                    ftol=float(xnes_params['ftol']),
                    xtol=float(xnes_params['xtol']),
                    memory=xnes_params['memory'],
                    force_bounds=xnes_params['force_bounds'])
    return uda


def init_moead(moead_params):
    #moead_params = algo_config['moead']
    uda = pg.moead(gen=1,
                     weight_generation=moead_params['weight_generation'],
                     decomposition=moead_params['decomposition'],
                     neighbours=moead_params['neighbours'],
                     CR=moead_params['CR'],
                     F=moead_params['F'],
                     eta_m=moead_params['eta_m'],
                     realb=moead_params['realb'],
                     limit=moead_params['limit'],
                     preserve_diversity=moead_params['preserve_diversity'])
    return uda