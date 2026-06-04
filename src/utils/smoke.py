from __future__ import annotations

from pprint import pprint

from .device import get_device, describe_device
from .seed import seed_everything


def run_smoke(device: str = "auto", seed: int = 42, deterministic: bool = False) -> None:
    """
    Internal smoke test for utility imports + device resolution.
    """
    seed_everything(seed=seed, deterministic=deterministic)
    resolved = get_device(device)
    info = describe_device(resolved)

    print("Smoke test OK")
    pprint(
        {
            "seed": seed,
            "deterministic": deterministic,
            "requested_device": device,
            "resolved_device": str(resolved),
            "device_info": info,
        }
    )