from enum import Enum
from typing import Optional

from ollama import chat
from pydantic import BaseModel, Field

from camera_test import describe_image

MODEL_NAME = "gemma3:4b"


class PathStatus(str, Enum):
    CLEAR = "clear"
    BLOCKED = "blocked"
    UNCERTAIN = "uncertain"


class VisualSceneAssessment(BaseModel):
    path_status: PathStatus
    obstacle_present: bool
    obstacle_description: Optional[str]
    central_region_description: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


SYSTEM_PROMPT = """
You classify a drone's forward path based on a free-text scene
description of a single forward-facing drone image.

Rules:
1. Use only evidence stated in the description.
2. Do not claim metric depth or distance.
3. Use "clear" when the description indicates open, navigable ground
   space in the forward path, even if buildings, parked vehicles, or
   people are present to the sides or in the background.
4. Use "blocked" only when the description indicates a substantial
   object (vehicle, person, wall, barrier) directly occupying the
   navigable path itself, not merely visible nearby or at the edges
   of the scene.
5. Use "uncertain" when the description is ambiguous or does not give
   enough information about the navigable path to decide.
6. Confidence must be between 0 and 1.
7. This is an offline semantic assessment, not a flight-safety decision.
"""


def assess_image(image_path: str) -> VisualSceneAssessment:
    description = describe_image(image_path)

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Scene description:\n{description}\n\n"
                "Classify the forward path.",
            },
        ],
        format=VisualSceneAssessment.model_json_schema(),
        options={"temperature": 0, "num_ctx": 4096},
    )
    return VisualSceneAssessment.model_validate_json(
        response.message.content
    )


def main() -> None:
    image_path = input("Image path: ").strip()
    try:
        result = assess_image(image_path)
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
