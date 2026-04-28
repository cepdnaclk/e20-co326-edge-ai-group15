import math
import random
import time


def generate_vibration() -> float:
    """Generate a single healthy motor vibration reading in g-units."""
    now = time.time()
    simulated_frequency_hz = 10.0
    phase = 2 * math.pi * simulated_frequency_hz * now
    amplitude = random.uniform(0.1, 0.3)
    baseline = amplitude * abs(math.sin(phase))

    noise = random.gauss(0, 0.02)
    reading = baseline + noise
    return round(max(0.0, reading), 4)