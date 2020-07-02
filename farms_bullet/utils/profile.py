"""Profile"""

import pstats
import cProfile


def profile(function, **kwargs):
    """Profile with cProfile"""
    n_time = kwargs.pop('pstat_n_time', 30)
    n_cumtime = kwargs.pop('pstat_n_cumtime', 30)
    prof = cProfile.Profile()
    profile_filename = kwargs.pop('profile_filename', 'simulation.profile')
    result = prof.runcall(function, **kwargs)
    prof.dump_stats(profile_filename)
    pstat = pstats.Stats(profile_filename)
    pstat.sort_stats('time').print_stats(n_time)
    pstat.sort_stats('cumtime').print_stats(n_cumtime)
    return result
