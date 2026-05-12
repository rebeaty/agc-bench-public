# HELM Scenario Template

Reference for generating HELM-compliant Scenario classes.

## Basic Structure

```python
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from datasets import load_dataset

class MyScenario(Scenario):
    name = "my_benchmark"
    description = "source/dataset-name"
    tags = ["creativity"]

    def get_instances(self, output_path):
        dataset = load_dataset("source/dataset", split="test")

        instances = []
        for item in dataset:
            # Format prompt inline
            prompt = f"Question: {item['question']}"

            # Build references inline (see patterns below)
            references = [Reference(Output(text=item['answer']), tags=[CORRECT_TAG])]

            instances.append(Instance(
                input=Input(text=prompt),
                references=references,
                split=TEST_SPLIT
            ))
        return instances
```

## Reference Patterns

### Multiple Choice
ALL choices become References. Only correct one gets CORRECT_TAG.

```python
references = []
for i, choice in enumerate(choices):
    letter = chr(65 + i)  # A, B, C, D
    is_correct = (i == correct_index)
    tags = [CORRECT_TAG] if is_correct else []
    references.append(Reference(Output(text=letter), tags=tags))
```

### Binary (Yes/No)
Both options are References.

```python
references = [
    Reference(Output(text="Yes"), tags=[CORRECT_TAG] if label == 1 else []),
    Reference(Output(text="No"), tags=[CORRECT_TAG] if label == 0 else [])
]
```

### Single Answer
```python
references = [Reference(Output(text=str(answer)), tags=[CORRECT_TAG])]
```

### Open-Ended (No Correct Answer)
```python
references = []  # Empty is fine for divergent thinking tasks
```

## Key Rules

1. `name` = lowercase with underscores
2. `description` = data source reference (NOT task description)
3. `tags` = include "creativity" plus relevant category
4. Always use `CORRECT_TAG` for correct answers
5. Always specify `split` (usually `TEST_SPLIT`)

## Multimodal Scenarios

For benchmarks with images, audio, or video inputs.

### Imports for Multimodal

```python
from helm.benchmark.scenarios.scenario import (
    Scenario, Instance, Input, Output, Reference,
    CORRECT_TAG, TEST_SPLIT
)
from helm.common.media_object import MediaObject, MultimediaObject
from datasets import load_dataset
```

### Pattern: Vision-Language Task

```python
class MyVisionScenario(Scenario):
    name = "my_vision_benchmark"
    description = "source/dataset-name"
    tags = ["creativity", "multimodal", "vision"]

    def get_instances(self, output_path):
        dataset = load_dataset("source/dataset", split="test")

        instances = []
        for item in dataset:
            # Create multimedia content (text + image)
            multimedia_content = MultimediaObject([
                MediaObject(
                    content_type="text/plain",
                    text=f"Question: {item['question']}"
                ),
                MediaObject(
                    content_type="image/jpeg",
                    location=item['image_path']  # or URL
                )
            ])

            # Build references (same as text-only)
            references = [
                Reference(
                    Output(text=choice),
                    tags=[CORRECT_TAG] if choice == item['answer'] else []
                )
                for choice in item['choices']
            ]

            instances.append(Instance(
                input=Input(multimedia_content=multimedia_content),
                references=references,
                split=TEST_SPLIT
            ))
        return instances
```

### Pattern: Audio Task

```python
multimedia_content = MultimediaObject([
    MediaObject(
        content_type="text/plain",
        text="Transcribe or analyze this audio:"
    ),
    MediaObject(
        content_type="audio/mp3",
        location="/path/to/audio.mp3"  # or URL
    )
])

instance = Instance(
    input=Input(multimedia_content=multimedia_content),
    references=[...],
    split=TEST_SPLIT
)
```

### Pattern: Multiple Images

```python
multimedia_content = MultimediaObject([
    MediaObject(content_type="text/plain", text="Compare these images:"),
    MediaObject(content_type="image/png", location=image1_path),
    MediaObject(content_type="image/png", location=image2_path),
    MediaObject(content_type="text/plain", text="\nWhich shows more creativity?")
])
```

### Supported Content Types

- **Images**: `image/png`, `image/jpeg`, `image/gif`, `image/webp`
- **Audio**: `audio/mp3`, `audio/wav`, `audio/ogg`
- **Video**: `video/mp4`, `video/webm`
- **Text**: `text/plain` (for inline text within multimedia)

### Text vs Multimodal Decision

**Use text-only** (traditional `Input(text=...)`) when:
- Visual elements can be serialized as text (coordinates, grids, etc.)
- Task doesn't require actual media processing
- Evaluating text-only models

**Use multimodal** (`Input(multimedia_content=...)`) when:
- Task requires visual/audio understanding
- Benchmarking vision-language or audio-language models
- Media content is essential to the task
