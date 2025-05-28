# Genetic-Flappy-AI

## Overview
This project implements Flappy Bird with two modes:
- **Manual Play:** Play Flappy Bird yourself using the keyboard.
- **Genetic Algorithm (GA) Mode:** Watch AI birds learn to play Flappy Bird by evolving their neural networks with a genetic algorithm.

## Installation
1. Clone this repository.
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually:
   ```bash
   pip install pygame numpy
   ```

## How to Run

### Play Manually
Run:
```bash
python flappy_game.py
```
- Press **Space** to make the bird jump.
- On 'Game Over', press **R** to restart.

### Run the Genetic Algorithm (AI)
Run:
```bash
python flappy_ga.py [flags]
```

#### Available Flags for `flappy_ga.py`:
- `--no-checkpoint`      
  Do **not** load checkpointed neural network weights (start fresh).
- `--population_size N`  
  Set the number of birds per generation (default: 100).
- `--injection_rate F`
  Fraction (0–1) of the new population to inject with the checkpoint network if loaded (default: 0.2).

**Examples:**
- Default (with checkpointing):
  ```
  python flappy_ga.py
  ```
- Run with 200 birds and no checkpoint:
  ```
  python flappy_ga.py --population_size 200 --no-checkpoint
  ```

## Checkpoints
- The AI saves the best neural network’s weights in `checkpoint.npz`.
- Use checkpoints to resume or improve AI training later.

## Requirements
- Python 3.x
- pygame, numpy

---
Enjoy evolving and playing Flappy Bird!
