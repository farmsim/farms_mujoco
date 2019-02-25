"""Evolution - Test phase computation"""

import pygmo as pg

import numpy as np
import matplotlib.pyplot as plt

from salamander_evolution.archipelago import ArchiEvolution


class SineModel:
    """SineModel"""

    def __init__(self, amplitude, frequency, phase, offset):
        super(SineModel, self).__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    @classmethod
    def from_vector(cls, vector):
        """From vector"""
        return cls(*vector)

    def to_vector(self):
        """To vector"""
        return np.array([
            self.amplitude,
            self.frequency,
            self.phase,
            self.offset
        ])

    def data(self, xdata):
        """Data"""
        return (
            self.amplitude*np.sin(
                2*np.pi*self.frequency*xdata
                + self.phase
            ) + self.offset
        )

    def plot(self, label, start, end, style="-"):
        """Plot"""
        xdata = np.linspace(start, end, 3e3)
        plt.plot(xdata, self.data(xdata), style, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Data")
        plt.legend()
        plt.grid(True)


class NSineModel:
    """SineModel"""

    def __init__(self, sines):
        super(NSineModel, self).__init__()
        self.sines = sines

    @classmethod
    def from_vector(cls, vector):
        """From vector"""
        assert len(vector) % 4 == 0
        return cls([
            SineModel.from_vector(vector[4*i:4*i+4])
            for i in range(len(vector)//4)
        ])

    def to_vector(self):
        """To vector"""
        return np.concatenate([
            sine.to_vector()
            for sine in self.sines
        ])

    def data(self, xdata):
        """Data"""
        return np.sum([sine.data(xdata) for sine in self.sines], axis=0)

    def plot(self, label, start, end, style="-"):
        """Plot"""
        xdata = np.linspace(start, end, 3e3)
        plt.plot(xdata, self.data(xdata), style, label=label)
        plt.xlabel("Time [s]")
        plt.ylabel("Data")
        plt.legend()
        plt.grid(True)


class SineFitting:
    """SineFitting"""

    def __init__(self, sine, xdata):
        super(SineFitting, self).__init__()
        self._sine = sine
        self.xdata = xdata
        self._data = (
            sine.data(xdata)
            + np.random.normal(0, 0.1*self._sine.amplitude, np.size(xdata))
        )
        self._name = "Sine fitting"
        self._max_freq = 0.5*len(xdata)/(xdata[-1] - xdata[0])

    def fitness(self, decision_vector):
        """Fitness"""
        fit_sine = SineModel.from_vector(decision_vector)
        fit_data = fit_sine.data(self.xdata)
        return [np.mean(np.square(self._data - fit_data))]
        # + 1e-3*(fit_sine.frequency**2)

    def get_name(self):
        """Get name"""
        return self._name

    def get_bounds(self):
        """Get bounds"""
        return (
            [0, 0, 0, -100],
            [100, self._max_freq, 2*np.pi, 100],
        )

    def best_known(self):
        """Best known"""
        return self._sine.to_vector()


class NSineFitting:
    """NSineFitting"""

    def __init__(self, nsines, xdata):
        super(NSineFitting, self).__init__()
        self._nsines = nsines
        self._size = len(nsines.sines)
        self.xdata = xdata
        self._data = (
            nsines.data(xdata)
            # + np.random.normal(0, 0.1*self._nsines.amplitude, np.size(xdata))
        )
        self._name = "Nsines fitting"
        self._max_freq = 0.5*len(xdata)/(xdata[-1] - xdata[0])

    def fitness(self, decision_vector):
        """Fitness"""
        fit_nsines = NSineModel.from_vector(decision_vector)
        fit_data = fit_nsines.data(self.xdata)
        return [np.mean(np.square(self._data - fit_data))]
        # + 1e-3*(fit_nsines.frequency**2)

    def get_name(self):
        """Get name"""
        return self._name

    def get_bounds(self):
        """Get bounds"""
        return (
            np.concatenate([[0, 0, 0, -100] for _ in range(self._size)]),
            np.concatenate([
                [100, self._max_freq, 2*np.pi, 100]
                for _ in range(self._size)
            ])
        )

    def best_known(self):
        """Best known"""
        return self._nsines.to_vector()


def main():
    """Main"""

    problem_size = 2

    # Original data
    amplitudes = [10*np.random.ranf() for _ in range(problem_size)]
    freqs = [10*np.random.ranf() for _ in range(problem_size)]
    phases = [2*np.pi*np.random.ranf() for _ in range(problem_size)]
    offsets = [10*np.random.ranf() for _ in range(problem_size)]
    models = [
        SineModel(amplitudes[i], freqs[i], phases[i], offsets[i])
        for i in range(problem_size)
    ]
    nmodels = NSineModel(models)
    xdata = np.linspace(0, 3, 100)

    # Model
    kwargs = {"memory": True, "seed": 0}
    n_threads = 4
    algorithms = [pg.sade(gen=1, **kwargs) for _ in range(n_threads)]

    # Optimisation problem
    problem = NSineFitting(nmodels, xdata)

    # Evolution
    print("Running evolution")
    evolution = ArchiEvolution(
        problem=problem,
        algorithms=algorithms,
        n_pop=10,
        n_gen=300
    )

    # Result
    print("Evolution complete, getting result")
    champion = evolution.champion()
    fit_model = NSineModel.from_vector(champion[0])
    # champion = evolution.champion()
    # fit_model = SineModel.from_vector(champion[0])
    # message = (
    #     "Errors (fitness={}):"
    #     "\n  Amplitude: {} %"
    #     "\n  Frequency: {} %"
    #     "\n  Phase: {}"
    #     "\n  Offset: {}"
    # )
    # print(message.format(
    #     champion[1],
    #     100*np.abs((fit_model.amplitude - model.amplitude) / model.amplitude),
    #     100*np.abs((fit_model.frequency - model.frequency) / model.frequency),
    #     np.abs(model.phase - fit_model.phase) % (2*np.pi),
    #     np.abs(model.offset - fit_model.offset)
    # ))
    nmodels.plot("Original data", xdata[0], xdata[-1], ".-")
    fit_model.plot("Fitted data", xdata[0], xdata[-1], "-")
    plt.show()


if __name__ == '__main__':
    main()
