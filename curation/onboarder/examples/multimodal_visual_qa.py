"""
HELM Scenario: Multimodal Visual QA Example

This is a template/example for creating multimodal scenarios with images.
Demonstrates how to use MediaObject and MultimediaObject for vision-language tasks.

Pattern:
  - Question text followed by image
  - Multiple choice answer format
  - Images referenced by file path or URL

Key imports:
  - MediaObject: Individual media items (text, image, audio, video)
  - MultimediaObject: Container for sequences of media items

Multimodal content structure:
  MultimediaObject([
      MediaObject(content_type="text/plain", text="Question text"),
      MediaObject(content_type="image/jpeg", location="path/or/url")
  ])

Replace this example with actual dataset when onboarding a real benchmark.
"""

from typing import List
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Input,
    Reference,
    Output,
    CORRECT_TAG,
    TEST_SPLIT,
)
from helm.common.media_object import MediaObject, MultimediaObject


class MultimodalVisualQAScenario(Scenario):
    """
    Example multimodal scenario showing vision-language pattern.

    Replace this with actual dataset loading when onboarding a real benchmark.
    """

    name = "multimodal_visual_qa_example"
    description = "example/multimodal-visual-qa"
    tags = ["creativity", "multimodal", "vision"]

    def __init__(self, use_local_images: bool = True):
        """
        Args:
            use_local_images: If True, use local file paths; if False, use URLs
        """
        super().__init__()
        self.use_local_images = use_local_images

    def get_instances(self, output_path: str) -> List[Instance]:
        """
        Creates multimodal instances with image + text inputs.

        In a real scenario, replace this with:
            dataset = load_dataset("org/dataset-name", split="test")
            for item in dataset:
                # Extract image_path, question, choices, answer from item
        """

        # Example data structure (replace with actual dataset)
        example_data = [
            {
                "id": "example_1",
                "image_path": "/path/to/image1.jpg",  # or use URL
                "image_url": "https://example.com/image1.jpg",
                "question": "What creative technique is demonstrated in this artwork?",
                "choices": ["Surrealism", "Impressionism", "Abstract Expressionism", "Cubism"],
                "answer": "Surrealism"
            },
            {
                "id": "example_2",
                "image_path": "/path/to/image2.jpg",
                "image_url": "https://example.com/image2.jpg",
                "question": "Which emotion does this composition primarily evoke?",
                "choices": ["Joy", "Melancholy", "Anger", "Serenity"],
                "answer": "Melancholy"
            }
        ]

        instances = []
        for item in example_data:
            # Choose between local file path or URL
            image_location = item['image_path'] if self.use_local_images else item['image_url']

            # Create multimedia content: text question + image
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=f"Question: {item['question']}\n\nChoices:"
                ),
                MediaObject(
                    content_type="image/jpeg",  # or image/png, image/gif, etc.
                    location=image_location
                ),
                MediaObject(
                    content_type="text/plain",
                    text="\n" + "\n".join(
                        f"{chr(65+i)}) {choice}"
                        for i, choice in enumerate(item['choices'])
                    ) + "\n\nAnswer:"
                )
            ])

            # Build references: all choices, correct one tagged
            references = []
            for i, choice in enumerate(item['choices']):
                letter = chr(65 + i)  # A, B, C, D
                is_correct = (choice == item['answer'])
                tags = [CORRECT_TAG] if is_correct else []
                references.append(
                    Reference(Output(text=letter), tags=tags)
                )

            # Create instance with multimodal input
            instances.append(
                Instance(
                    input=Input(multimedia_content=multimedia_content),
                    references=references,
                    split=TEST_SPLIT,
                    id=item['id']
                )
            )

        return instances


# Alternative patterns for different multimodal scenarios:

def example_pattern_audio():
    """Pattern for audio-based tasks"""
    multimedia_content = MultimediaObject([
        MediaObject(
            content_type="text/plain",
            text="Listen to this audio and answer: What emotion is expressed?"
        ),
        MediaObject(
            content_type="audio/mp3",
            location="/path/to/audio.mp3"  # or URL
        )
    ])
    return multimedia_content


def example_pattern_multiple_images():
    """Pattern for comparing multiple images"""
    multimedia_content = MultimediaObject([
        MediaObject(
            content_type="text/plain",
            text="Compare these two artworks. Which shows more creative use of color?"
        ),
        MediaObject(
            content_type="image/png",
            location="/path/to/image_a.png"
        ),
        MediaObject(
            content_type="text/plain",
            text="vs."
        ),
        MediaObject(
            content_type="image/png",
            location="/path/to/image_b.png"
        ),
        MediaObject(
            content_type="text/plain",
            text="\nAnswer: A or B?"
        )
    ])
    return multimedia_content


def example_pattern_video():
    """Pattern for video understanding tasks"""
    multimedia_content = MultimediaObject([
        MediaObject(
            content_type="text/plain",
            text="Watch this video and describe the creative narrative technique used."
        ),
        MediaObject(
            content_type="video/mp4",
            location="/path/to/video.mp4"  # or URL
        )
    ])
    return multimedia_content


# Content type reference:
# Images: image/png, image/jpeg, image/gif, image/webp
# Audio: audio/mp3, audio/wav, audio/ogg
# Video: video/mp4, video/webm
# Text: text/plain (for inline text within multimedia sequence)
