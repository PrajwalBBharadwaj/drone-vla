from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from prototype import (
    DroneMotionCommand,
    LateralDirection,
    LongitudinalDirection,
    YawDirection,
    interpret_command,
)
from vision_test import (
    MonocularDepthAssessment,
    PathStatus,
    VisualSceneAssessment,
    assess_image,
    assess_monocular_depth,
)


UNCERTAIN_BLOCK_CONFIDENCE_THRESHOLD = 0.9
DEPTH_CLEARANCE_OVERRIDE_THRESHOLD = 0.58
DEPTH_CENTER_NEAR_FRACTION_MAX = 0.35
# Depth can only override a blocked call if semantic confidence is below this.
# At or above this value the semantic result wins regardless of depth.
DEPTH_OVERRIDE_MAX_CONFIDENCE = 0.90
# Max speed (m/s) allowed when depth overrides a blocked scene (constrained path).
CONSTRAINED_MAX_SPEED_MPS = 2.0
LATERAL_CLEARANCE_OVERRIDE_THRESHOLD = 0.55
LATERAL_NEAR_FRACTION_MAX = 0.40
LATERAL_CONSTRAINED_MAX_SPEED_MPS = 1.5
AUTO_YAW_CLEARANCE_DEADBAND = 0.05
AUTO_YAW_GAIN = 40.0
AUTO_YAW_MIN_SPEED_DEG_S = 5.0
AUTO_YAW_MAX_SPEED_DEG_S = 30.0


@dataclass
class IntegratedDecision:
    requested_command: DroneMotionCommand
    command: DroneMotionCommand
    scene: Optional[VisualSceneAssessment]
    depth: Optional[MonocularDepthAssessment]
    path_recommendation: Optional[dict]
    override_reason: Optional[str]


def _format_decision_summary(decision: IntegratedDecision) -> str:
    req = decision.requested_command
    applied = decision.command

    requested_forward = req.longitudinal_direction == LongitudinalDirection.FORWARD
    applied_forward = applied.longitudinal_direction == LongitudinalDirection.FORWARD

    if requested_forward and not applied_forward:
        return (
            "Decision: FORWARD suppressed "
            f"({req.longitudinal_speed_mps:.1f} -> {applied.longitudinal_speed_mps:.1f} m/s)"
        )

    if requested_forward and applied_forward:
        if applied.longitudinal_speed_mps != req.longitudinal_speed_mps:
            return (
                "Decision: FORWARD accepted with speed cap "
                f"({req.longitudinal_speed_mps:.1f} -> {applied.longitudinal_speed_mps:.1f} m/s)"
            )
        return f"Decision: FORWARD accepted ({applied.longitudinal_speed_mps:.1f} m/s)"

    requested_lateral = req.lateral_direction != LateralDirection.NONE
    applied_lateral = applied.lateral_direction != LateralDirection.NONE
    if requested_lateral and not applied_lateral:
        return (
            "Decision: LATERAL suppressed "
            f"({req.lateral_speed_mps:.1f} -> {applied.lateral_speed_mps:.1f} m/s)"
        )

    if requested_lateral and applied_lateral:
        if applied.lateral_speed_mps != req.lateral_speed_mps:
            return (
                "Decision: LATERAL accepted with speed cap "
                f"({req.lateral_speed_mps:.1f} -> {applied.lateral_speed_mps:.1f} m/s)"
            )
        return (
            "Decision: LATERAL accepted "
            f"({applied.lateral_direction.value} at {applied.lateral_speed_mps:.1f} m/s)"
        )

    return "Decision: Non-forward command applied"


def _uncertain_has_strong_block_evidence(scene: VisualSceneAssessment) -> bool:
    if scene.path_status != PathStatus.UNCERTAIN:
        return False
    if scene.confidence < UNCERTAIN_BLOCK_CONFIDENCE_THRESHOLD:
        return False

    evidence_text = " ".join(
        [
            scene.explanation,
            scene.obstacle_description or "",
            scene.central_region_description,
        ]
    ).lower()
    strong_block_terms = (
        "directly in front",
        "across the path",
        "across the road",
        "entire path",
        "completely blocked",
        "blocked off",
        "no passage",
        "impassable",
        "barrier",
        "barricade",
        "wall",
    )
    return any(term in evidence_text for term in strong_block_terms)


def _depth_supports_forward(depth: MonocularDepthAssessment) -> bool:
    return (
        depth.center_clearance_score >= DEPTH_CLEARANCE_OVERRIDE_THRESHOLD
        and depth.center_near_obstacle_fraction <= DEPTH_CENTER_NEAR_FRACTION_MAX
    )


def _depth_supports_lateral(
    depth: MonocularDepthAssessment,
    direction: LateralDirection,
) -> bool:
    if direction == LateralDirection.LEFT:
        return (
            depth.left_clearance_score >= LATERAL_CLEARANCE_OVERRIDE_THRESHOLD
            and depth.left_near_obstacle_fraction <= LATERAL_NEAR_FRACTION_MAX
        )
    if direction == LateralDirection.RIGHT:
        return (
            depth.right_clearance_score >= LATERAL_CLEARANCE_OVERRIDE_THRESHOLD
            and depth.right_near_obstacle_fraction <= LATERAL_NEAR_FRACTION_MAX
        )
    return True


def _should_suppress_forward(
    scene: VisualSceneAssessment,
    depth: Optional[MonocularDepthAssessment],
) -> bool:
    if scene.path_status == PathStatus.BLOCKED:
        if (
            depth is not None
            and _depth_supports_forward(depth)
            and scene.confidence < DEPTH_OVERRIDE_MAX_CONFIDENCE
        ):
            return False
        return True

    return _uncertain_has_strong_block_evidence(scene)


def _apply_speed_cap(
    command: DroneMotionCommand,
    scene: VisualSceneAssessment,
    depth: Optional[MonocularDepthAssessment],
) -> tuple:
    """Return (capped_command, cap_reason | None)."""
    is_constrained = (
        scene.path_status == PathStatus.BLOCKED
        or (scene.obstacle_present and scene.path_status == PathStatus.UNCERTAIN)
    )
    if not is_constrained:
        return command, None
    updates = {}
    reasons = []

    if command.longitudinal_speed_mps > CONSTRAINED_MAX_SPEED_MPS:
        updates["longitudinal_speed_mps"] = CONSTRAINED_MAX_SPEED_MPS
        reasons.append(
            f"forward speed capped to {CONSTRAINED_MAX_SPEED_MPS} m/s"
        )

    if (
        command.lateral_direction != LateralDirection.NONE
        and command.lateral_speed_mps > LATERAL_CONSTRAINED_MAX_SPEED_MPS
    ):
        updates["lateral_speed_mps"] = LATERAL_CONSTRAINED_MAX_SPEED_MPS
        reasons.append(
            f"lateral speed capped to {LATERAL_CONSTRAINED_MAX_SPEED_MPS} m/s"
        )

    if not updates:
        return command, None

    capped = command.model_copy(update=updates)
    reason = (
        "; ".join(reasons)
        + f": constrained path (path_status={scene.path_status.value}, "
        + f"confidence={scene.confidence:.2f})"
    )
    return capped, reason


def _should_suppress_lateral(
    command: DroneMotionCommand,
    scene: VisualSceneAssessment,
    depth: Optional[MonocularDepthAssessment],
) -> bool:
    if command.lateral_direction == LateralDirection.NONE:
        return False

    if depth is not None and not _depth_supports_lateral(depth, command.lateral_direction):
        return True

    # Extremely high-confidence blocked scenes are treated as no-go even laterally.
    if scene.path_status == PathStatus.BLOCKED and scene.confidence >= 0.95:
        return True

    return False


def _apply_auto_yaw_from_clearance(
    command: DroneMotionCommand,
    depth: Optional[MonocularDepthAssessment],
    requested_command: DroneMotionCommand,
) -> tuple:
    """Return (command_with_yaw, yaw_reason|None) based on depth clearance."""
    if depth is None:
        return command, None
    if command.longitudinal_direction != LongitudinalDirection.FORWARD:
        return command, None

    # Respect explicit user yaw commands.
    if requested_command.yaw_direction != YawDirection.NONE:
        return command, "Auto-yaw skipped: user provided explicit yaw command."

    clearance_delta = depth.right_clearance_score - depth.left_clearance_score
    if abs(clearance_delta) < AUTO_YAW_CLEARANCE_DEADBAND:
        command = command.model_copy(
            update={
                "yaw_direction": YawDirection.NONE,
                "yaw_speed_deg_s": 0.0,
            }
        )
        return command, None

    yaw_direction = (
        YawDirection.CLOCKWISE
        if clearance_delta > 0
        else YawDirection.COUNTERCLOCKWISE
    )
    yaw_speed = min(
        AUTO_YAW_MAX_SPEED_DEG_S,
        max(AUTO_YAW_MIN_SPEED_DEG_S, abs(clearance_delta) * AUTO_YAW_GAIN),
    )
    command = command.model_copy(
        update={
            "yaw_direction": yaw_direction,
            "yaw_speed_deg_s": round(yaw_speed, 2),
        }
    )
    return (
        command,
        "Auto-yaw applied from clearance "
        f"(left={depth.left_clearance_score:.3f}, "
        f"right={depth.right_clearance_score:.3f}, "
        f"delta={clearance_delta:+.3f}): "
        f"{yaw_direction.value} at {yaw_speed:.1f} deg/s",
    )


def plan_motion(user_text: str, image_path: Optional[str]) -> IntegratedDecision:
    command = interpret_command(user_text)

    wants_safety_check = (
        command.valid
        and (
            command.longitudinal_direction == LongitudinalDirection.FORWARD
            or command.lateral_direction != LateralDirection.NONE
        )
    )
    if not wants_safety_check or not image_path:
        return IntegratedDecision(
            requested_command=command,
            command=command,
            scene=None,
            depth=None,
            path_recommendation=None,
            override_reason=None,
        )

    requested_command = command
    scene = assess_image(image_path)
    depth = assess_monocular_depth(image_path)

    safety_reasons = []

    if _should_suppress_forward(scene, depth):
        command = command.model_copy(
            update={
                "longitudinal_direction": LongitudinalDirection.NONE,
                "longitudinal_speed_mps": 0.0,
            }
        )
        safety_reasons.append(
            f"Forward motion suppressed: path_status={scene.path_status.value} "
            f"({scene.explanation})"
        )

    if _should_suppress_lateral(command, scene, depth):
        command = command.model_copy(
            update={
                "lateral_direction": LateralDirection.NONE,
                "lateral_speed_mps": 0.0,
            }
        )
        safety_reasons.append(
            "Lateral motion suppressed: insufficient side clearance or scene is "
            "high-confidence blocked."
        )

    command, yaw_reason = _apply_auto_yaw_from_clearance(
        command,
        depth,
        requested_command,
    )
    if yaw_reason:
        safety_reasons.append(yaw_reason)

    command, cap_reason = _apply_speed_cap(command, scene, depth)
    if cap_reason:
        safety_reasons.append(cap_reason)

    return IntegratedDecision(
        requested_command=requested_command,
        command=command,
        scene=scene,
        depth=depth,
        path_recommendation=None,
        override_reason="; ".join(safety_reasons) if safety_reasons else None,
    )


def main() -> None:
    print("Integrated command + vision prototype")
    print("No drone is connected.")
    print("Type 'quit' to exit.\n")

    last_image_path: Optional[str] = None

    while True:
        user_text = input("Command> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue

        image_prompt = "Image path"
        if last_image_path:
            image_prompt += f" [{last_image_path}]"
        image_path = input(f"{image_prompt}> ").strip() or last_image_path
        last_image_path = image_path or last_image_path

        try:
            decision = plan_motion(user_text, image_path)
            print(_format_decision_summary(decision))
            print("Requested command:")
            print(decision.requested_command.model_dump_json(indent=2))
            print("\nApplied command:")
            print(decision.command.model_dump_json(indent=2))
            if decision.scene is not None:
                print(f"\nScene: {decision.scene.model_dump_json(indent=2)}")
            if decision.depth is not None:
                print(f"\nDepth: {decision.depth.model_dump_json(indent=2)}")
            if decision.override_reason:
                print(f"\nSafety action: {decision.override_reason}")
            print()
        except ValidationError as error:
            print(f"Schema validation failed:\n{error}\n")
        except FileNotFoundError as error:
            print(f"{error}\n")
        except Exception as error:
            print(f"Model call failed:\n{error}\n")


if __name__ == "__main__":
    main()
