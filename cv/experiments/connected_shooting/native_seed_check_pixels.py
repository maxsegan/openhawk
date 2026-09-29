"""Describe check-pixel use by native net seeds without changing numerical work."""

from copy import deepcopy


def annotate(receipt: dict, reserved_check_frames=None) -> dict:
    """Bind the actual net/wing rays to the caller's original check partition."""
    result = deepcopy(receipt)
    net_frame = receipt.get("incoming_chart", {}).get("native_frame", receipt.get("net_frame"))
    wing = receipt.get("original_frames")
    known = net_frame is not None and wing is not None and reserved_check_frames is not None
    consumed = (
        sorted({float(net_frame), *map(float, wing)})
        if net_frame is not None and wing is not None
        else []
    )
    ground_frames = receipt.get("ground_support", {}).get("witness", {}).get("native_frames", [])
    consumed = sorted(set(consumed) | set(map(float, ground_frames)))
    check = sorted(set(consumed) & set(map(float, reserved_check_frames))) if known else None
    result["check_pixel_usage"] = dict(
        status="bound" if known else "unknown_partition_or_ray_receipt",
        net_native_frame=net_frame,
        wing_native_frames=wing,
        ground_witness_native_frames=ground_frames,
        source_frames_consumed=consumed,
        reserved_check_frames_consumed=check,
        uses_reserved_check_pixels=bool(check) if known else None,
        numerical_inputs_changed=False,
        scope="native net/wing initializer rays only; other refinement stages separate",
    )
    return result


def fit_usage(fit: dict) -> dict | None:
    receipt = fit.get("net_response_initialization")
    if receipt is None:
        return None
    return receipt.get("check_pixel_usage") or dict(
        status="unknown_legacy_native_seed_receipt",
        reserved_check_frames_consumed=None,
        uses_reserved_check_pixels=None,
    )


def selection_fields(candidates: list[dict]) -> dict:
    """Include losing contenders: their check-informed fits entered selection."""
    usages = [
        usage
        for row in candidates
        if (usage := fit_usage(row.get("measurement", {}).get("fit", {}))) is not None
    ]
    return selection_from_usages(usages)


def selection_from_usages(usages: list[dict]) -> dict:
    """Combine consumed-ray receipts, including searches without checkpoints."""
    known_used = any(u.get("uses_reserved_check_pixels") is True for u in usages)
    unknown = any(u.get("uses_reserved_check_pixels") is None for u in usages)
    used = True if known_used else None if unknown else False
    return dict(
        selector_uses_withheld_pixels=used,
        selector_uses_withheld_pixels_directly=False,
        native_seed_check_pixel_usage=dict(
            completed_native_contenders=len(usages),
            check_informed_contenders=sum(
                u.get("uses_reserved_check_pixels") is True for u in usages
            ),
            unknown_partition_contenders=sum(
                u.get("uses_reserved_check_pixels") is None for u in usages
            ),
            reserved_check_frames_consumed=sorted(
                {float(f) for u in usages for f in (u.get("reserved_check_frames_consumed") or [])}
            ),
            independent_check_pixels_preserved=(None if used is None else not used),
            scope="indirect native-seed use in contenders; rank still uses its original training objective",
        ),
    )


def refinement_disclosures(receipt: dict) -> list[dict]:
    """Disclose actual all-native additions alongside legacy stage accounting."""

    def epochs(value):
        if value is None:
            return set()
        if isinstance(value, (list, tuple)):
            return {frame for row in value for frame in epochs(row)}
        return {float(value)}

    rows = []
    for stage in receipt.get("stages", []):
        fit = stage.get("fit") or {}
        policy = (fit.get("fit_policy") or {}).get("heldout")
        reported = stage.get("consumed_check_rows")
        added = fit.get("full_native_rows_added")
        if "consumed_check_rows" in stage or added is not None or policy is not None:
            reported_epochs, added_epochs = epochs(reported), epochs(added)
            rows.append(
                dict(
                    stage=stage.get("stage"),
                    status=stage.get("status"),
                    consumed_check_rows=deepcopy(reported),
                    consumed_check_rows_scope="legacy stage field; may omit all-native additions",
                    full_native_rows_added=deepcopy(added),
                    consumed_check_native_frames=sorted(reported_epochs | added_epochs),
                    all_native_additions_missing_from_legacy_field=sorted(
                        added_epochs - reported_epochs
                    ),
                    existing_check_policy=policy,
                    scope="refinement consumption; distinct from native seed search",
                )
            )
    return rows
