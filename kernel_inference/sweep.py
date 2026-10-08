"""Expand a sweep spec (workflow/config/*.yml) into one configuration per model.

Options merge from the spec's blocks, later ones winning: defaults, noise variant, representation
arm, kernel, then the panel's `set`, `kernel`, `noise` and `gate` blocks, then every matching
`overrides` entry. The random-feature kernel expands over the panel's `rff_gamma` grid.

    python -m kernel_inference.sweep workflow/config/paper.yml   # one JSON configuration per line
"""
import json
import sys

import yaml


def expand(spec):
    configs = []
    for panel, p in spec["panels"].items():
        for variant, v in spec["variants"].items():
            for arm in p["arms"]:
                for kernel in p["kernels"]:
                    for gamma in p["rff_gamma"] if kernel == "rff" else [None]:
                        where = {"panel": panel, "variant": variant, "arm": arm, "kernel": kernel}
                        stem = kernel if gamma is None else f"rff_g{repr(float(gamma)).removesuffix('.0').replace('.', 'p')}_d256"
                        c = {"fit": f"{variant}_{arm}/{panel}/{stem}", "panel": panel, **spec["defaults"], **v,
                             **spec["arms"][arm], **spec["kernels"][kernel], **p.get("set", {}),
                             **p.get("kernel", {}).get(kernel, {}), **p.get("noise", {}).get(v["noise_model"], {}),
                             **(p.get("gate", {}) if variant in ("gaussian_gate", "laplace_gate") else {})}
                        for o in spec.get("overrides", []):
                            if all(where[k] == x for k, x in o["where"].items()):
                                c.update(o["set"])
                        configs.append(c if gamma is None else {**c, "rff_gamma": gamma})
    return configs


def load(path):
    return expand(yaml.safe_load(open(path)))


if __name__ == "__main__":
    for c in load(sys.argv[1]):
        print(json.dumps(c))
