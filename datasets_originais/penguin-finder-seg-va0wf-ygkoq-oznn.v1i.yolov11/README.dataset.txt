# penguin-finder-seg-va0wf-ygkoq-oznn > 2025-03-13 8:59am
https://universe.roboflow.com/rf100-vl/penguin-finder-seg-va0wf-ygkoq-oznn

Provided by a Roboflow user
License: MIT

# Overview
- [Introduction](#introduction)
- [Object Classes](#object-classes)
  - [Penguin](#penguin)

# Introduction
The Penguin Finder dataset is designed to detect and identify penguins in various environments. The dataset focuses on a single class:

- **Penguin**: Includes any penguin visible in the image, characterized by its distinctive shape and posture.

# Object Classes

## Penguin
### Description
Penguins are flightless birds characterized by a distinctive upright stance, flippers, and a contrasting black and white coloration. Their head is usually black with a contrasting white belly, and they often have bright orange or yellow markings on their neck or beak.

### Instructions
- Annotate the full body of the penguin, including its head, body, flippers, and feet, as long as they are visible.
- If the penguin's features are partially occluded by objects or other penguins, include the estimate of the occluded areas in the annotation box.
- Do not annotate if the penguin is less than 20% visible or if it is unclear what the object is due to extreme blurriness.
- Ensure the bounding box is tight around the penguin, following the contour of the body and flippers, but allow slight margin where exact boundaries are ambiguous.
- Ignore shadows, reflections, or any other non-penguin elements in the image.
- If there are multiple penguins in proximity, ensure each penguin has its own distinct bounding box without overlap.