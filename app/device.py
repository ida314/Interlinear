"""Where the model stages run.

Settings holds a device *preference* as a plain string; this module turns it into a concrete
torch device. The split matters because `app.config` instantiates `settings` at import time,
and importing torch at import time would undo the project's rule that every heavy dependency
is imported inside the stage that needs it.

An explicit preference is never second-guessed: if you asked for "cpu" you get "cpu", and
torch is not imported at all.
"""

from __future__ import annotations

AUTO = "auto"


def resolve(pref: str = AUTO) -> str:
    """Map a device preference onto something torch can accept.

    `auto` walks cuda -> mps -> cpu. Anything else is returned verbatim, so an explicit
    preference never silently becomes something else.
    """
    if pref != AUTO:
        return pref
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        # The pure stages (segment, plan, render) work without torch; let them.
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str):
    """Half precision only where it is both supported and a win.

    fp16 on MPS has produced NaN logits in Whisper's decoder across several torch releases,
    and MPS would not deliver the speedup anyway, so it stays fp32.
    """
    import torch  # noqa: PLC0415

    return torch.float16 if device.startswith("cuda") else torch.float32


def describe(pref: str = AUTO) -> str:
    """One line for logs and /health, naming the actual hardware where there is any."""
    device = resolve(pref)
    if not device.startswith("cuda"):
        return device
    try:
        import torch  # noqa: PLC0415

        return f"{device} ({torch.cuda.get_device_name(0)})"
    except Exception:  # noqa: BLE001
        return device
