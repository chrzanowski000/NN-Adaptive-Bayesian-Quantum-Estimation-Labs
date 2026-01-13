import numpy as np
from modules.algorithms.seq_montecarlo import (
    build_model,
    init_particles,
    smc_update_no_resample,
    ess,
)
from modules.simulation import FIXED_T2


def measure_once(true_omega, t, rng):
    """
    Single Bernoulli measurement.
    """
    p0 = (
        np.exp(-t / FIXED_T2)
        * np.cos(true_omega * t / 2)**2
        + (1 - np.exp(-t / FIXED_T2)) / 2
    )
    return 0 if rng.random() < p0 else 1


def test_single_update():
    rng = np.random.default_rng(0)

    # ----------------------------
    # Ground truth + setup
    # ----------------------------
    TRUE_OMEGA = 0.7
    t = 2.0              # INFORMATIVE TIME
    N = 2000

    model, logp_fn = build_model()
    particles, logw = init_particles(N)

    # ----------------------------
    # Before update
    # ----------------------------
    print("Before update:")
    print("  ESS =", ess(logw))
    print("  logw std =", np.std(logw))

    # ----------------------------
    # One observation
    # ----------------------------
    d = measure_once(TRUE_OMEGA, t, rng)
    print("\nObserved d =", d)

    particles, logw = smc_update_no_resample(
        particles, logw, d, t, model, logp_fn
    )

    # ----------------------------
    # After update
    # ----------------------------
    print("\nAfter update:")
    print("  ESS =", ess(logw))
    print("  logw std =", np.std(logw))

    # ----------------------------
    # HARD assertions
    # ----------------------------
    assert np.std(logw) > 0, "❌ logw did not change"
    assert ess(logw) < N, "❌ ESS did not drop"
    print("\n✅ SMC UPDATE WORKS")


if __name__ == "__main__":
    test_single_update()