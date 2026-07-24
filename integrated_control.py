from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from prototype import DroneMotionCommand, LongitudinalDirection, interpret_command
from vision_test import PathStatus, VisualSceneAssessment, assess_image


@dataclass
class IntegratedDecision:
    command: DroneMotionCommand
    scene: Optional[VisualSceneAssessment]
    override_reason: Optional[str]


def plan_motion(user_text: str, image_path: Optional[str]) -> IntegratedDecision:
    command = interpret_command(user_text)

    wants_forward = (
        command.valid
        and command.longitudinal_direction == LongitudinalDirection.FORWARD
    )
    if not wants_forward or not image_path:
        return IntegratedDecision(command=command, scene=None, override_reason=None)

    scene = assess_image(image_path)

    if scene.path_status in (PathStatus.BLOCKED, PathStatus.UNCERTAIN):
        command = command.model_copy(
            update={
                "longitudinal_direction": LongitudinalDirection.NONE,
                "longitudinal_speed_mps": 0.0,
            }
        )
        override_reason = (
            f"Forward motion suppressed: path_status={scene.path_status.value} "
            f"({scene.explanation})"
        )
        return IntegratedDecision(
            command=command, scene=scene, override_reason=override_reason
        )

    return IntegratedDecision(command=command, scene=scene, override_reason=None)


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
            print(decision.command.model_dump_json(indent=2))
            if decision.scene is not None:
                print(f"\nScene: {decision.scene.model_dump_json(indent=2)}")
            if decision.override_reason:
                print(f"\nOverride: {decision.override_reason}")
            print()
        except ValidationError as error:
            print(f"Schema validation failed:\n{error}\n")
        except FileNotFoundError as error:
            print(f"{error}\n")
        except Exception as error:
            print(f"Model call failed:\n{error}\n")


if __name__ == "__main__":
    main()
