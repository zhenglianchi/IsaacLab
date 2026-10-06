"""Tensor-only ORU phase logic, independent of simulator state and reward method."""
import torch


def update_phase(
    phase, entry_count, contact, contact_count, stage_alpha, contact_alpha,
    xy_error, angle_error, height_gap, contact_force, *, cfg, dt,
):
    """Guidance is height-scheduled; the contact stage is force-driven.

    ``phase`` (guidance) still uses height on purpose: it only schedules when the
    virtual reference starts its 2 cm/s descent towards the seat, and that has to
    begin at the pre-insert height or nothing brings the part down in a controlled
    way. The stage that matters for the reward blend and the force cap is the
    CONTACT one, and that is now decided by the filtered ORU<->ground contact force
    alone - no height gate. The wrist reaction is unusable for this: free-space
    chain inertia reads 3.9-5.9 N and the seated bounce reads 2-10 N, so the bands
    overlap and the old logic confirmed "contact" 82 mm above the seat.

    This is a task sequencer, not a classifier of true peg/hole contact.
    """
    # 2026-10-06 (user request): the guidance stage and the descent permission are no
    # longer gated by the ATTITUDE, and the +-5 mm height window at the pre-insert point
    # is gone. Reason: the gate checked the PART attitude while the controller only
    # regulates the FLANGE, so a residual twist about the tool axis (0.1-0.38 rad, see
    # the handover entry) made the gate unsatisfiable -> phase stayed 0 -> the reference
    # held the part at the pre-insert height and the arm hovered, motionless, until the
    # 90 s timeout (97.7% of C0 failures). Blocking the descent 80 mm above the seat was
    # also against this task's own design rule that switching is contact-driven: the part
    # is now allowed to descend and CONTACT decides what happens next.
    # in_corridor = laterally above the docking axis; the attitude requirement is kept
    # only for reporting (aligned) and for the success criterion, not as a precondition.
    aligned = (xy_error < cfg.entry_xy_tolerance) & (angle_error < cfg.entry_angle_tolerance)
    in_corridor = xy_error < cfg.entry_xy_tolerance
    entry_count = torch.where(in_corridor, entry_count + 1, 0)
    phase = phase | (entry_count >= cfg.entry_confirm_steps)
    phase = phase & (height_gap <= cfg.preinsert_height + 2 * cfg.entry_height_tolerance)
    can_advance = phase & (height_gap > -cfg.seat_z_tolerance)
    # Real ORU<->ground contact force (zero while airborne), debounced by
    # entry_confirm_steps and released only below contact_leave_threshold.
    # Contact is decided by force alone (plus the lateral corridor), NOT by attitude:
    # otherwise an unachievable attitude gate also kills the contact stage.
    touching = in_corridor & (contact_force > cfg.contact_force_threshold)
    contact_count = torch.where(touching, contact_count + 1, 0)
    contact = (contact | (contact_count >= cfg.entry_confirm_steps)) & in_corridor & (
        contact_force >= cfg.contact_leave_threshold
    )
    delta = min(1.0, dt / cfg.switch_duration_s)
    # Reward blend AND the Z force cap follow the contact stage (the original design
    # intent: contact detection drives both). Blending the insertion reward in from
    # the pre-insert height instead taxed the whole last 8 cm with the insertion
    # costs, which is what taught the policy to avoid the insertion stage.
    stage_alpha = stage_alpha + (contact.float() - stage_alpha).clamp(-delta, delta)
    contact_alpha = contact_alpha + (contact.float() - contact_alpha).clamp(-delta, delta)
    return phase, entry_count, contact, contact_count, stage_alpha, contact_alpha, can_advance


def axial_reference(current, ee_z, phase, can_advance, contact_alpha, *, seat_z, cfg, dt):
    """Rate-limit a virtual spring anchor; the physical seat stays unchanged."""
    # 2026-10-06: single-target-point mode. The reference is the seat pose itself, so the
    # controller receives one target point instead of a trajectory - the classic
    # fixed-impedance baseline, which is what excites the outward arc and the
    # tool-axis clocking drift that the staged version avoids.
    if getattr(cfg, "reference_mode", "ramp") == "setpoint":
        # NB: contact_alpha is a tensor, so build the result by broadcasting against
        # current instead of torch.full_like (which needs a scalar fill value).
        return (seat_z - cfg.insertion_bias * contact_alpha) + torch.zeros_like(current)
    desired = torch.where(phase, seat_z - cfg.insertion_bias * contact_alpha, seat_z + cfg.preinsert_height)
    # On loss of alignment do not move the spring anchor farther down. Release
    # existing preload gradually; no abrupt target jump or forced tilt.
    desired = torch.where(phase & ~can_advance, torch.maximum(ee_z, current), desired)
    return current + (desired - current).clamp(-cfg.reference_speed * dt, cfg.reference_speed * dt)


def seat_geometry_ok(xy_error, height_gap, angle_error, *, cfg):
    """Position/attitude half of the seat test, WITHOUT the velocity gates.

    Used to decide when to stop controlling: once the ORU is geometrically in the
    seat, residual control force is what makes it bounce off again, and the bounce
    is what keeps the velocity gates (speed / angular speed) unsatisfied. Freezing
    on the geometry lets the motion decay, after which the full seat test can hold.
    """
    return (
        (xy_error < cfg.xy_tolerance)
        & (height_gap.abs() < cfg.seat_z_tolerance)
        & (angle_error < cfg.seat_angle_tolerance)
    )


def seat_candidate(xy_error, height_gap, angle_error, speed, angular_speed, *, cfg):
    """A bounded geometric tolerance rejects both incomplete and over-deep seats."""
    return (
        seat_geometry_ok(xy_error, height_gap, angle_error, cfg=cfg)
        & (speed < cfg.success_speed_tolerance)
        & (angular_speed < cfg.success_angular_speed_tolerance)
    )
