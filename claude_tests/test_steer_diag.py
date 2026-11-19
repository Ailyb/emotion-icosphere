"""Model-free test for probekit_steer_diag.diagnose."""
import numpy as np

from probekit_steer_diag import diagnose


def main():
    rng = np.random.default_rng(0); H = 20
    ax = np.zeros(H); ax[0] = 1.0
    X = rng.normal(size=(300, H)); y = np.array(["neutral"] * 150 + ["joy"] * 150, dtype=object); X[150:] += 3 * ax
    along = ax.copy(); ortho = np.zeros(H); ortho[1] = 1.0
    r = diagnose(X, y, {"a": np.stack([along]), "b": np.stack([ortho]), "random": rng.normal(size=(5, H))}, ["joy"])
    assert r["a"]["joy"][0] > 0.95 and abs(r["b"]["joy"][0]) < 0.3 and abs(r["random"]["joy"][0]) < 0.5
    print("test_steer_diag: OK")


if __name__ == "__main__":
    main()
