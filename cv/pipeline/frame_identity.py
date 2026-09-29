"""Parse extracted-frame identities without a fixed filename-width assumption."""

from pathlib import Path


def frame_number_from_name(value: str | Path) -> int:
    """Read the entire nonnegative index in an ``f_<digits>`` image name.

    Four-digit padding is a minimum display width, not a maximum frame index.
    Reject malformed identities instead of silently mapping them to another frame.
    """
    stem = Path(value).stem
    digits = stem[2:] if stem.startswith("f_") else ""
    if not digits or not digits.isascii() or not digits.isdecimal():
        raise ValueError(f"invalid extracted-frame identity: {value!s}")
    return int(digits)
