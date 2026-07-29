from enum import Enum
from typing import Optional

from ollama import chat
from pydantic import BaseModel, Field, ValidationError, model_validator

MODEL_NAME = "gemma3:4b"


class LongitudinalDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    NONE = "none"


class LateralDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class VerticalDirection(str, Enum):
    UP = "up"
    DOWN = "down"
    NONE = "none"


class YawDirection(str, Enum):
    CLOCKWISE = "clockwise"
    COUNTERCLOCKWISE = "counterclockwise"
    NONE = "none"


class DroneMotionCommand(BaseModel):
    valid: bool
    longitudinal_direction: LongitudinalDirection
    longitudinal_speed_mps: float = Field(ge=0.0)
    lateral_direction: LateralDirection
    lateral_speed_mps: float = Field(ge=0.0)
    vertical_direction: VerticalDirection
    vertical_speed_mps: float = Field(ge=0.0)
    yaw_direction: YawDirection
    yaw_speed_deg_s: float = Field(ge=0.0)
    duration_seconds: Optional[float] = Field(default=None, gt=0.0)
    reason: Optional[str] = None

    @model_validator(mode="after")
    def validate_semantics(self) -> "DroneMotionCommand":
        if self.valid and self.reason is not None:
            raise ValueError("A valid command must not include a rejection reason.")
        if not self.valid and not self.reason:
            raise ValueError("An invalid command must include a reason.")

        axes = [
            (self.longitudinal_direction, LongitudinalDirection.NONE,
             self.longitudinal_speed_mps, "longitudinal"),
            (self.lateral_direction, LateralDirection.NONE,
             self.lateral_speed_mps, "lateral"),
            (self.vertical_direction, VerticalDirection.NONE,
             self.vertical_speed_mps, "vertical"),
            (self.yaw_direction, YawDirection.NONE,
             self.yaw_speed_deg_s, "yaw"),
        ]

        for direction, none_value, speed, axis_name in axes:
            if direction == none_value and speed != 0.0:
                raise ValueError(
                    f"{axis_name} speed must be zero when direction is none."
                )
            if direction != none_value and speed <= 0.0:
                raise ValueError(
                    f"{axis_name} speed must be positive when motion is requested."
                )
        return self


SYSTEM_PROMPT = """
You are a strict natural-language motion-command interpreter for a drone.

Convert the user's request into exactly one DroneMotionCommand.

Rules:
1. Extract only motion explicitly requested by the user.
2. Never add movement on an axis that the user did not mention.
3. For every unrequested axis, use direction "none" and speed 0.0.
4. Translational speeds must be in meters per second.
5. Yaw speeds must be in degrees per second.
6. Convert units only when the conversion is unambiguous.
7. Never invent a numerical speed.
8. Words such as quickly, slowly, fast, and gently do not define a
   numerical speed.
9. Multiple explicitly requested motions may be active simultaneously.
10. Stop, hover, and hold mean zero motion on all axes.
11. Forward means toward the direction in which the drone's nose points.
12. Do not determine joystick signs.
13. If the request is ambiguous, incomplete, unrelated, or unsupported:
    set valid false, set all directions to none, all speeds to 0.0,
    and include a concise reason.
14. If valid is true, reason must be null.
15. Return only the structured response.

Examples:

User: Climb at one meter per second.
Output: vertical up at 1.0 m/s; every other axis none at 0.0.

User: Move left at 2 m/s while rotating clockwise at 15 degrees per second.
Output: lateral left at 2.0 m/s and yaw clockwise at 15.0 deg/s;
every other axis none at 0.0.

User: Go forward quickly.
Output: invalid because no numerical forward speed was specified.
"""


def interpret_command(user_text: str) -> DroneMotionCommand:
    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        format=DroneMotionCommand.model_json_schema(),
        options={"temperature": 0, "num_ctx": 4096},
    )
    command = DroneMotionCommand.model_validate_json(response.message.content)

    text = user_text.lower()
    mentions_sideways = "sideways" in text or "sideway" in text
    has_explicit_lateral_direction = (
        "left" in text
        or "right" in text
        or "port" in text
        or "starboard" in text
    )

    if (
        mentions_sideways
        and not has_explicit_lateral_direction
        and command.valid
        and command.lateral_direction != LateralDirection.NONE
    ):
        return command.model_copy(
            update={
                "valid": False,
                "longitudinal_direction": LongitudinalDirection.NONE,
                "longitudinal_speed_mps": 0.0,
                "lateral_direction": LateralDirection.NONE,
                "lateral_speed_mps": 0.0,
                "vertical_direction": VerticalDirection.NONE,
                "vertical_speed_mps": 0.0,
                "yaw_direction": YawDirection.NONE,
                "yaw_speed_deg_s": 0.0,
                "duration_seconds": None,
                "reason": (
                    "Ambiguous lateral motion: specify 'left' or 'right' when using "
                    "'sideways'."
                ),
            }
        )

    return command


def main() -> None:
    print("Gemma drone-command prototype")
    print("No drone is connected.")
    print("Type 'quit' to exit.\n")

    while True:
        user_text = input("Command> ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        try:
            result = interpret_command(user_text)
            print(result.model_dump_json(indent=2))
            print()
        except ValidationError as error:
            print(f"Schema validation failed:\n{error}\n")
        except Exception as error:
            print(f"Model call failed:\n{error}\n")


if __name__ == "__main__":
    main()
