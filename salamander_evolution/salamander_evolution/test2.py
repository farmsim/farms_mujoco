import pygmo as pg
from simulation_subroutine import evol_problem

def main():
    prob = pg.problem(
        evol_problem(
            dim=1,
            link_name="link_body_0",
            path=".gazebo/models/"
        )
    )
    res = []
    n_pop = 5
    gen = 1
    pop = pg.population(prob, n_pop)
    print(pop)
    isl = pg.island(
        algo=pg.pso(gen),
        prob=prob,
        size=n_pop,
        udi=pg.mp_island()
    )
    for i in range(10):
        print("Creating island")
        # islands.append(isl)
        # print("Populations:\n===\n{}\n===\n{}".format(pop, isl.get_population()))
        print("Setting population")
        # isl.set_population(pop)
        print("EVOLVING!! ({})".format(i))
        isl.evolve()
        isl.wait()
        print("EVOLVING DONE!!")
        res.append(isl.get_population().champion_f[0])
        print(isl)
        # pop = isl.get_population()
        # print(pop)

    import matplotlib.pyplot as plt 
    plt.plot(res) 
    plt.show()

if __name__ == '__main__':
    main()
