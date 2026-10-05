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
    aligned = (xy_error < cfg.entry_xy_tolerance) & (angle_error < cfg.entry_angle_tolerance)
    at_entry = aligned & ((height_gap - cfg.preinsert_height).abs() < cfg.entry_height_tolerance)
    entry_count = torch.where(at_entry, entry_count + 1, 0)
    phase = phase | (entry_count >= cfg.entry_confirm_steps)
    phase = phase & (height_gap <= cfg.preinsert_height + 2 * cfg.entry_height_tolerance)
    can_advance = aligned & phase & (height_gap > -cfg.seat_z_tolerance)
    # Real ORU<->ground contact force (zero while airborne), debounced by
    # entry_confirm_steps and released only below contact_leave_threshold.
    touching = aligned & (contact_force > cfg.contact_force_threshold)
    contact_count = torch.where(touching, contact_count + 1, 0)
    contact = (contact | (contact_count >= cfg.entry_confirm_steps)) & aligned & (
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
