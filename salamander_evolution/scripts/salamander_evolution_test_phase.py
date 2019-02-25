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

    def to_vector(self):
        """To vector"""
        return np.array([
            self.amplitude,
            self.frequency,
            self.phase,
            self.offset
        ])

    @classmethod
    def from_vector(cls, vector):
        """From vector"""
        return cls(*vector)

    def data(self, xdata):
        """Data"""
        return (
            self.amplitude*np.sin(
                2*np.pi*self.frequency*xdata
                + self.phase
            )
            + self.offset
        )

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
        """Fitnesss"""
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


def main():
    """Main"""
    # Original data
    amplitude = 10*np.random.ranf()
    freq = 10*np.random.ranf()
    phase = 2*np.pi*np.random.ranf()
    offset = 10*np.random.ranf()
    model = SineModel(amplitude, freq, phase, offset)
    xdata = np.linspace(0, 3, 100)

    # Model
    # amplitude*sin(2*np.pi*frequency + phase)
    # Parameters: amplitude, frequency, phase
    kwargs = {"memory": True, "seed": 0}
    n_threads = 10
    algorithms = [pg.sade(gen=1, **kwargs) for _ in range(n_threads)]

    # Optimisation problem
    problem = SineFitting(model, xdata)

    # Evolution
    print("Running evolution")
    evolution = ArchiEvolution(
        problem=problem,
        algorithms=algorithms,
        n_pop=10,
        n_gen=100
    )

    # Result
    print("Evolution complete, getting result")
    champion = evolution.champion()
    fit_model = SineModel.from_vector(champion[0])
    message = (
        "Errors (fitness={}):"
        "\n  Amplitude: {}"
        "\n  Frequency: {}"
        "\n  Phase: {}"
        "\n  Offset: {}"
    )
    print(message.format(
        champion[1],
        np.abs(model.amplitude - fit_model.amplitude),
        np.abs(model.frequency - fit_model.frequency),
        np.abs(model.phase - fit_model.phase) % (2*np.pi),
        np.abs(model.offset - fit_model.offset)
    ))
    model.plot("Original data", xdata[0], xdata[-1], ".-")
    fit_model.plot("Fitted data", xdata[0], xdata[-1], "-")
    plt.show()


if __name__ == '__main__':
    main()
