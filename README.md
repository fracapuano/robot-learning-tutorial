# Robot Learning: A Tutorial

This repository contains the source code for the "Robot Learning: A Tutorial" report. This tutorial covers many of the most pressing aspects in modern robot learning, and provides practice examples using `lerobot`, the robot-learning library developed by Hugging Face.

You’re more than welcome to contribute to the next edition of the tutorial!
Simply open an issue, tag @fracapuano, and start a discussion about the scope and content you’d like to add. Check out CONTRIBUTING.md for more details 😊
All merged pull requests will receive public acknowledgment in the main body of the tutorial.
Items marked with an empty `[ ]` in the following Table of Contents are open for community contribution!

## Table of Contents

### 1. Introduction
- [x] 1.1 `lerobot` Dataset
    - [x] 1.1.1 The dataset class design
- [x] 1.2 Code Example: Batching a (Streaming) Dataset
- [x] 1.3 Code Example: Collecting Data

### 2. Classical Robotics
- [x] 2.1 Explicit and Implicit Models
- [x] 2.2 Different Types of Motion
- [x] 2.3 Example: Planar Manipulation
    - [x] 2.3.1 Adding Feedback Loops
- [x] 2.4 Limitations of Dynamics-based Robotics

### 3. Robot (Reinforcement) Learning
- [x] 3.1 A (Concise) Introduction to RL
- [x] 3.2 Real-world RL for Robotics
- [x] 3.3 Code Example: Real-world RL
- [x] 3.4 Limitations of RL in Real-World Robotics: Simulators and Reward Design

### 4. Robot (Imitation) Learning
- [x] 4.1 A (Concise) Introduction to Generative Models
    - [x] 4.1.1 Variational Auto-Encoders
    - [x] 4.1.2 Diffusion Models
    - [x] 4.1.3 Flow Matching
- [x] 4.2 Action Chunking with Transformers
    - [x] 4.2.1 Code Example: Training and Using ACT in Practice
- [x] 4.3 Diffusion Policy
    - [x] 4.3.1 Code Example: Training and Using Diffusion Policies in Practice
- [x] 4.4 Optimized Inference
    - [x] 4.4.1 Code Example: Using Async Inference

### 5. Generalist Robot Policies
- [x] 5.1 Preliminaries: Models and Data
- [x] 5.2 Modern VLAs
    - [x] 5.2.1 VLMs for VLAs
- [x] 5.3 PI0
    - [ ] 5.3.1 Code Example: Using PI0
- [x] 5.4 SmolVLA
    - [ ] 5.4.1 Code Example: Using SmolVLA
- [ ] 5.5 GR00T (1/2)
    - [ ] 5.5.1 Code Example: Using GR00T
- [ ] 5.6 PI05
    - [ ] 5.6.1 Code Example: Using PI05
- [ ] Large-scale datasets
    - [ ] Open-X
    - [ ] DROID
    - [ ] BEHAVIOR

### 6. Some Emerging Directions in Robot Learning
- [ ] 6.1 Post training VLAs
    - [ ] 6.1.1 From Imitation to Refinement
    - [ ] 6.1.2 EXPO
- [ ] 6.2 World Models for robotics
    - [ ] 6.2.1 Cosmos
    - [ ] 6.2.2 World Models (1X)
    - [ ] 6.2.3 Sima and Genie 1

### 7. Conclusions
- [x] 7.1 Conclusions

## License

The written content of this book is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).

All source code examples in the `snippets/` directory are licensed under the [MIT License](https://opensource.org/licenses/MIT).
