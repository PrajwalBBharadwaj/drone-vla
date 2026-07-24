from pathlib import Path
from ollama import chat

MODEL_NAME = "gemma3:4b"


def describe_image(image_path: str) -> str:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    response = chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": (
                    "Describe the important objects, open areas, buildings, "
                    "vegetation, people, vehicles, and possible navigable "
                    "space visible in this image. Be concise."
                ),
                "images": [str(path)],
            }
        ],
        options={"temperature": 0, "num_ctx": 4096},
    )
    return response.message.content


def main() -> None:
    image_path = input("Image path: ").strip()
    try:
        print("\nGemma response:\n")
        print(describe_image(image_path))
    except Exception as error:
        print(f"\nError: {error}")


if __name__ == "__main__":
    main()

