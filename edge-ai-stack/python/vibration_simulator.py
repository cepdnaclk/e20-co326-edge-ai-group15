"""Vibration signal simulator for a motor condition monitoring pipeline"""

import math
import random
import time
from collections.abc import Generator


def generate_vibration(fault: bool = False) -> float:
    """Generate a single motor vibration reading in g-units.

    In normal mode, the signal approximates a healthy rotating machine with a
    low amplitude sinusoidal oscillation and small sensor noise.

    In fault mode, the signal emulates bearing/imbalance events either as sharp
    transient spikes or as high-energy irregular burst noise.

    Args:
        fault: If True, generate fault-like vibration behavior.

    Returns:
        A single vibration reading in g.
    """
    if fault:
        fault_mode = random.choice(["spike", "burst"])
        if fault_mode == "spike":
            # Physical intent: abrupt impulse caused by impact/friction fault.
            return round(random.uniform(0.8, 2.0), 4)

        # Physical intent: unstable resonance like burst with large random energy.
        burst_center = random.uniform(0.9, 1.4)
        burst_noise = random.gauss(0, 0.25)
        return round(max(0.8, min(2.0, burst_center + burst_noise)), 4)

    # Physical intent: baseline periodic vibration from rotating shaft dynamics.
    now = time.time()
    simulated_frequency_hz = 10.0
    phase = 2 * math.pi * simulated_frequency_hz * now
    amplitude = random.uniform(0.1, 0.3)
    baseline = amplitude * abs(math.sin(phase))

    # Sensor/environment noise around the nominal periodic signal.
    noise = random.gauss(0, 0.02)
    reading = baseline + noise
    return round(max(0.0, reading), 4)


def simulate_stream() -> Generator[float, None, None]:
    """Yield an infinite stream of vibration readings at 1-second intervals.

    Faults are randomly injected with approximately 15% probability.
    """
    while True:
        is_fault = random.random() < 0.15
        yield generate_vibration(fault=is_fault)
        time.sleep(1)



