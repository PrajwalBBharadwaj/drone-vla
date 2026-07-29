from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
from ollama import chat
from PIL import Image
from pydantic import BaseModel, Field

try:
    from transformers import pipeline
except ImportError:  # pragma: no cover - handled with runtime error message
    pipeline = None

from camera_test import describe_image

MODEL_NAME = "gemma3:4b"
DEPTH_MODEL_NAME = "Intel/dpt-hybrid-midas"
_DEPTH_PIPELINE = None


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


class MonocularDepthAssessment(BaseModel):
    model_name: str
    center_clearance_score: float = Field(ge=0.0, le=1.0)
    center_near_obstacle_fraction: float = Field(ge=0.0, le=1.0)
    left_clearance_score: float = Field(ge=0.0, le=1.0)
    left_near_obstacle_fraction: float = Field(ge=0.0, le=1.0)
    right_clearance_score: float = Field(ge=0.0, le=1.0)
    right_near_obstacle_fraction: float = Field(ge=0.0, le=1.0)
    global_near_obstacle_fraction: float = Field(ge=0.0, le=1.0)
    note: str


SYSTEM_PROMPT = """
You classify a drone's forward path based on a free-text scene
description of a single forward-facing drone image.

Assume a compact drone footprint (about 1m x 1m).

Rules:
1. Use only evidence stated in the description.
2. Do not claim metric depth or distance.
3. Use "clear" when the description indicates open, navigable ground
   space in the forward path, even if buildings, parked vehicles, or
   people are present to the sides or in the background.
4. Treat roads, streets, lanes, and open corridors as navigable by
    default unless an obstacle is explicitly described as directly
    occupying and closing the immediate forward route.
5. Use "blocked" only when the description indicates a substantial
   object (vehicle, person, wall, barrier) directly occupying the
   navigable path itself, not merely visible nearby or at the edges
   of the scene.
6. If people or vehicles are merely present in the scene without an
    explicit statement that they block the forward route, prefer
    "uncertain" over "blocked".
7. Use "uncertain" when the description is ambiguous or does not give
   enough information about the navigable path to decide.
8. Confidence must be between 0 and 1.
9. This is an offline semantic assessment, not a flight-safety decision.
"""


def _should_downgrade_blocked_to_uncertain(
    description: str, scene: VisualSceneAssessment
) -> bool:
    if scene.path_status != PathStatus.BLOCKED:
        return False

    text = description.lower()
    obstacle_text = (scene.obstacle_description or "").lower()
    central_text = scene.central_region_description.lower()

    street_context_terms = ("street", "road", "lane", "avenue", "corridor")
    has_street_context = any(
        term in text or term in central_text for term in street_context_terms
    )

    has_dynamic_actors = any(
        term in text or term in obstacle_text
        for term in ("vehicle", "vehicles", "pedestrian", "pedestrians", "traffic")
    )

    explicit_hard_block_terms = (
        "barricade",
        "barrier",
        "wall",
        "fence",
        "roadblock",
        "blocked off",
        "completely blocked",
        "occupying the entire",
        "across the path",
        "across the road",
        "directly in front",
    )
    has_explicit_hard_block = any(term in text for term in explicit_hard_block_terms)

    return has_street_context and has_dynamic_actors and not has_explicit_hard_block


def _get_depth_pipeline():
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is not None:
        return _DEPTH_PIPELINE
    if pipeline is None:
        raise RuntimeError(
            "transformers is not installed. Install dependencies in the virtual "
            "environment to enable monocular depth estimation."
        )

    _DEPTH_PIPELINE = pipeline("depth-estimation", model=DEPTH_MODEL_NAME)
    return _DEPTH_PIPELINE


def assess_monocular_depth(image_path: str) -> MonocularDepthAssessment:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    depth_pipeline = _get_depth_pipeline()
    image = Image.open(path).convert("RGB")
    output = depth_pipeline(image)

    predicted_depth = output.get("predicted_depth")
    if predicted_depth is None:
        raise RuntimeError("Depth model did not return predicted_depth.")

    depth_map = predicted_depth.squeeze().detach().cpu().numpy().astype(np.float32)
    if depth_map.ndim != 2:
        raise RuntimeError("Unexpected depth map dimensions.")

    finite_mask = np.isfinite(depth_map)
    if not finite_mask.any():
        raise RuntimeError("Depth map contains no finite values.")

    finite_values = depth_map[finite_mask]
    low, high = np.percentile(finite_values, [5, 95])
    denom = max(high - low, 1e-6)
    normalized_inverse_depth = np.clip((depth_map - low) / denom, 0.0, 1.0)

    height, width = normalized_inverse_depth.shape
    y0, y1 = height // 3, (2 * height) // 3
    x0, x1 = width // 3, (2 * width) // 3
    center_region = normalized_inverse_depth[y0:y1, x0:x1]
    left_region = normalized_inverse_depth[y0:y1, :x0]
    right_region = normalized_inverse_depth[y0:y1, x1:]

    near_threshold = 0.65
    center_near_fraction = float(np.mean(center_region >= near_threshold))
    left_near_fraction = float(np.mean(left_region >= near_threshold))
    right_near_fraction = float(np.mean(right_region >= near_threshold))
    global_near_fraction = float(np.mean(normalized_inverse_depth >= near_threshold))
    center_clearance_score = 1.0 - float(np.mean(center_region))
    left_clearance_score = 1.0 - float(np.mean(left_region))
    right_clearance_score = 1.0 - float(np.mean(right_region))

    return MonocularDepthAssessment(
        model_name=DEPTH_MODEL_NAME,
        center_clearance_score=round(center_clearance_score, 3),
        center_near_obstacle_fraction=round(center_near_fraction, 3),
        left_clearance_score=round(left_clearance_score, 3),
        left_near_obstacle_fraction=round(left_near_fraction, 3),
        right_clearance_score=round(right_clearance_score, 3),
        right_near_obstacle_fraction=round(right_near_fraction, 3),
        global_near_obstacle_fraction=round(global_near_fraction, 3),
        note=(
            "Relative monocular depth only: higher clearance score means a region "
            "looks farther/opener. This is not metric depth in meters."
        ),
    )


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
    scene = VisualSceneAssessment.model_validate_json(response.message.content)

    if _should_downgrade_blocked_to_uncertain(description, scene):
        scene = scene.model_copy(
            update={
                "path_status": PathStatus.UNCERTAIN,
                "confidence": min(scene.confidence, 0.7),
                "explanation": (
                    scene.explanation
                    + " Reclassified to uncertain because the scene suggests a "
                    "street-like corridor with nearby dynamic actors but no explicit "
                    "direct barrier across the forward route."
                ),
            }
        )

    return scene


def main() -> None:
    image_path = input("Image path: ").strip()
    try:
        result = assess_image(image_path)
        print(result.model_dump_json(indent=2))
    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
