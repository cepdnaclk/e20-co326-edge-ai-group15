"""vibration signal simulator for motor condition monitoring."""

import math
import random
import time
from collections import deque
from collections.abc import Callable, Generator


class MotorVibrationSimulator:
    def __init__(self, base_freq: float = 10.0, sample_rate: float = 1.0):
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.t = 0.0

    def _normal_signal(self) -> float:
        """Generate realistic normal vibration using harmonics + noise."""
        # Fundamental + harmonics
        A1 = random.uniform(0.15, 0.25)
        A2 = A1 * 0.5
        A3 = A1 * 0.3

        v = (
            A1 * math.sin(2 * math.pi * self.base_freq * self.t)
            + A2 * math.sin(2 * math.pi * 2 * self.base_freq * self.t)
            + A3 * math.sin(2 * math.pi * 3 * self.base_freq * self.t)
        )

        # Gaussian noise
        noise = random.gauss(0, 0.02)

        return abs(v + noise)

    def _fault_signal(self, base_value: float) -> float:
        """Apply fault patterns to base signal."""
        fault_type = random.choice(["spike", "burst", "imbalance", "drift"])

        if fault_type == "spike":
            # Sudden impulse
            return base_value + random.uniform(1.0, 2.5)

        elif fault_type == "burst":
            # Short high-energy noise
            return abs(random.gauss(1.5, 0.3))

        elif fault_type == "imbalance":
            # Amplitude modulation
            mod = 1 + 0.5 * math.sin(2 * math.pi * 1 * self.t)
            return abs(base_value * mod)

        elif fault_type == "drift":
            # Gradual increase
            drift_factor = 1 + 0.001 * self.t
            return abs(base_value * drift_factor)

        return base_value

    def generate(self, fault: bool = False) -> float:
        """Generate a single vibration reading."""
        base = self._normal_signal()

        if fault:
            value = self._fault_signal(base)
        else:
            value = base

        # Advance time
        self.t += 1.0 / self.sample_rate

        return round(max(0.0, min(3.0, value)), 4)


def simulate_stream(
    motor_running_fn: Callable[[], bool] | None = None,
) -> Generator[tuple[float, bool], None, None]:
    """Yield continuous (vibration, is_fault) stream."""
    simulator = MotorVibrationSimulator()

    while True:
        if motor_running_fn is not None and not motor_running_fn():
            yield (0.0, False)
            time.sleep(1)
            continue

        is_fault = random.random() < 0.15
        vibration = simulator.generate(fault=is_fault)

        yield (vibration, is_fault)
        time.sleep(1)