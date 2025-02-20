import pygame
import random
import sys
import numpy as np
import argparse

# ----------------------------------------------
# Import from the manual Flappy Bird script
# ----------------------------------------------
from flappy_game import (
    WIDTH, HEIGHT,
    WHITE, BLACK, GREEN, BLUE,
    FPS,
    PIPE_WIDTH, PIPE_GAP, PIPE_VELOCITY,
    BIRD_WIDTH, BIRD_HEIGHT,
    GRAVITY, JUMP_STRENGTH,
    create_pipe,
    FlappyGame
)

# ------------------ Configuration / GA Settings ------------------
POPULATION_SIZE = 100  # default number of birds per generation
MUTATION_RATE   = 0.1
INPUT_SIZE      = 4   # number of inputs fed to the neural network
HIDDEN_SIZE     = 8   # hidden neurons
OUTPUT_SIZE     = 1   # output: whether to jump or not

# ------------------ Neural Network ------------------
class NeuralNetwork:
    """
    A simple one-hidden-layer feed-forward network with a sigmoid output.
    """
    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE):
        """
        Initializes the neural network with random weights and biases.
        The hidden layer uses a tanh activation, and the output layer uses a sigmoid.
        """
        self.w1 = np.random.randn(hidden_size, input_size) * 0.5
        self.b1 = np.random.randn(hidden_size, 1) * 0.5
        self.w2 = np.random.randn(output_size, hidden_size) * 0.5
        self.b2 = np.random.randn(output_size, 1) * 0.5

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: A numpy column vector (shape: [input_size, 1]) of input features.
        :return: Output vector (shape: [output_size, 1]) after passing through the network.
        """
        # Hidden layer (tanh)
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.tanh(z1)
        # Output layer (sigmoid)
        z2 = np.dot(self.w2, a1) + self.b2
        output = 1 / (1 + np.exp(-z2))  # Sigmoid function
        return output

    def copy(self):
        """
        Returns a new NeuralNetwork instance with the exact same weights/biases.
        """
        nn = NeuralNetwork()
        nn.w1 = np.copy(self.w1)
        nn.b1 = np.copy(self.b1)
        nn.w2 = np.copy(self.w2)
        nn.b2 = np.copy(self.b2)
        return nn

    def mutate(self, rate=MUTATION_RATE):
        """
        Mutates the network's weights and biases by adding small random values
        with probability proportional to the mutation rate.
        """
        def mutate_array(arr):
            mutated = arr.copy()
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    if random.random() < rate:
                        mutated[i, j] += np.random.randn() * rate
            return mutated

        self.w1 = mutate_array(self.w1)
        self.b1 = mutate_array(self.b1)
        self.w2 = mutate_array(self.w2)
        self.b2 = mutate_array(self.b2)

    @staticmethod
    def crossover(parent1, parent2):
        """
        Creates a new child network by randomly choosing each weight/bias
        either from parent1 or parent2.
        """
        child = NeuralNetwork()

        def cross(mat1, mat2):
            shape = mat1.shape
            new_mat = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    new_mat[i, j] = random.choice([mat1[i, j], mat2[i, j]])
            return new_mat

        child.w1 = cross(parent1.w1, parent2.w1)
        child.b1 = cross(parent1.b1, parent2.b1)
        child.w2 = cross(parent1.w2, parent2.w2)
        child.b2 = cross(parent1.b2, parent2.b2)

        return child

# ------------------ Bird Class (Agent) ------------------
class Bird:
    """
    The Bird class represents an individual agent in the population.
    Each has a NeuralNetwork, position, velocity, and fitness metrics.
    """
    def __init__(self, brain=None):
        self.x = WIDTH // 4
        self.y = HEIGHT // 2
        self.velocity = 0
        self.alive = True
        self.fitness = 0
        self.score = 0
        self.passed_pipes = set()
        self.brain = brain if brain is not None else NeuralNetwork()

    def jump(self):
        """Make the bird jump (set upward velocity)."""
        self.velocity = JUMP_STRENGTH

    def update(self):
        """Update the bird's physics each frame and increase fitness slightly."""
        self.velocity += GRAVITY
        self.y += self.velocity
        self.fitness += 1  # each frame alive => +1 fitness

    def get_rect(self):
        return pygame.Rect(self.x, int(self.y), BIRD_WIDTH, BIRD_HEIGHT)

    def decide(self, pipe):
        """
        Decide whether to jump based on the closest pipe.
        Inputs (normalized):
          1) Bird's y
          2) Bird's velocity
          3) Horizontal distance to pipe
          4) Vertical difference from the gap center
        """
        if pipe is None:
            pipe_dist = 1.0
            gap_center = 0.5
        else:
            pipe_x = pipe.x
            pipe_dist = (pipe_x - self.x) / float(WIDTH)
            gap_center = (pipe.top_height + PIPE_GAP / 2) / float(HEIGHT)

        norm_y = self.y / float(HEIGHT)
        norm_vel = (self.velocity + 10) / 20.0
        diff = gap_center - norm_y

        inputs = np.array([[norm_y], [norm_vel], [pipe_dist], [diff]])
        output = self.brain.forward(inputs)
        if output[0][0] > 0.5:
            self.jump()

# ------------------ Pipe Class ------------------
class Pipe:
    """
    A single pair of pipes (top and bottom).
    Each pipe has an ID so birds can track scoring.
    """
    id_counter = 0

    def __init__(self):
        self.top_height = random.randint(50, HEIGHT - PIPE_GAP - 50)
        self.x = WIDTH
        self.id = Pipe.id_counter
        Pipe.id_counter += 1

    def update(self):
        """Move the pipe to the left."""
        self.x -= PIPE_VELOCITY

    def get_top_rect(self):
        return pygame.Rect(self.x, 0, PIPE_WIDTH, self.top_height)

    def get_bottom_rect(self):
        return pygame.Rect(
            self.x,
            self.top_height + PIPE_GAP,
            PIPE_WIDTH,
            HEIGHT - (self.top_height + PIPE_GAP)
        )

# ------------------ Population / GA Helpers ------------------
def next_generation(old_birds, population_size=POPULATION_SIZE):
    """
    Evolve the next generation of birds:
    1) Sort by fitness
    2) Keep best as elite
    3) Use fitness-proportional selection to pick parents
    4) Crossover + mutate
    """
    old_birds.sort(key=lambda b: b.fitness, reverse=True)
    print("Best fitness in generation:", old_birds[0].fitness)

    new_birds = []
    best = old_birds[0]
    new_birds.append(Bird(brain=best.brain.copy()))  # elitism

    fitness_sum = sum(b.fitness for b in old_birds)

    def select_parent():
        r = random.uniform(0, fitness_sum)
        running_sum = 0
        for bird in old_birds:
            running_sum += bird.fitness
            if running_sum > r:
                return bird
        return random.choice(old_birds)

    while len(new_birds) < population_size:
        parent1 = select_parent()
        parent2 = select_parent()
        child_brain = NeuralNetwork.crossover(parent1.brain, parent2.brain)
        child_brain.mutate()
        new_birds.append(Bird(brain=child_brain))

    return new_birds

# ------------------ Checkpoint Functions ------------------
CHECKPOINT_FILE = "checkpoint.npz"

def save_checkpoint(nn, fitness, filename=CHECKPOINT_FILE):
    """
    Saves the neural network weights and biases along with its fitness.
    """
    np.savez(filename, w1=nn.w1, b1=nn.b1, w2=nn.w2, b2=nn.b2, fitness=fitness)
    print("Checkpoint saved with fitness:", fitness)

def load_checkpoint(filename=CHECKPOINT_FILE):
    """
    Loads the checkpoint file and returns a tuple (neural network, fitness).
    If no file is found, returns (None, 0).
    """
    try:
        data = np.load(filename)
        nn = NeuralNetwork()
        nn.w1 = data["w1"]
        nn.b1 = data["b1"]
        nn.w2 = data["w2"]
        nn.b2 = data["b2"]
        fitness = float(data["fitness"])
        return nn, fitness
    except FileNotFoundError:
        return None, 0

# ------------------ GA-Flappy Game Class ------------------
class FlappyGAGame(FlappyGame):
    """
    Inherits from FlappyGame to reuse the basic setup,
    but overrides logic for multiple birds + GA.
    """
    def __init__(self, use_checkpoint=False, population_size=POPULATION_SIZE, injection_rate=0.0):
        # Override the caption to avoid "Flappy Bird (Manual)"
        super().__init__(caption="Flappy Bird (Genetic Algorithm Mode)")
        self.use_checkpoint = use_checkpoint
        self.population_size = population_size
        self.injection_rate = injection_rate

        # Always try to load the checkpoint file for comparison.
        loaded_nn, loaded_fitness = load_checkpoint(CHECKPOINT_FILE)
        if loaded_nn is not None:
            self.checkpoint_fitness = loaded_fitness
            print("Loaded checkpoint with fitness:", loaded_fitness)
            # Only load the weights into memory for use if the flag is specified.
            self.checkpoint_network = loaded_nn if self.use_checkpoint else None
        else:
            self.checkpoint_fitness = 0
            self.checkpoint_network = None

        self.generation = 1
        self.birds = [Bird() for _ in range(self.population_size)]

        self.last_checkpoint_pipe = 0

        # If we're using a checkpoint and a checkpoint network is available,
        # inject the checkpoint network into a fraction (injection_rate) of the population.
        if self.use_checkpoint and self.checkpoint_network is not None and self.injection_rate > 0:
            injection_count = int(round(self.population_size * self.injection_rate))
            indices = random.sample(range(self.population_size), injection_count)
            for idx in indices:
                self.birds[idx] = Bird(brain=self.checkpoint_network.copy())
            print(f"Injected checkpoint network into {injection_count} birds of the population.")

        self.pipes = [Pipe()]
        self.frame_count = 0
        self.best_lifetime_fitness = 0

    def draw_scoreboard(self):
        """
        Display additional GA stats on top-left:
        - Score (max among current birds)
        - Generation
        - # Alive
        - Current Gen Best fitness
        - Lifetime Best fitness
        """
        current_score = max(b.score for b in self.birds)
        lines = [
            f"Score: {current_score}",
            f"Generation: {self.generation}",
            f"Alive: {sum(b.alive for b in self.birds)}",
            f"Current Gen Best: {max(b.fitness for b in self.birds)}",
            f"Lifetime Best: {self.best_lifetime_fitness}"
        ]
        x, y = 10, 10
        line_gap = 20
        for line in lines:
            text_surf = self.font.render(line, True, WHITE)
            self.screen.blit(text_surf, (x, y))
            y += line_gap

    def run(self):
        """
        Main loop for GA-based Flappy Bird. 
        Multiple birds, when all die => next generation.
        """
        running = True
        while running:
            self.clock.tick(FPS)
            self.frame_count += 1

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()

            # Spawn new pipes
            if self.frame_count % 90 == 0:
                self.pipes.append(Pipe())

            # Update pipes
            for pipe in self.pipes:
                pipe.update()
            self.pipes = [pipe for pipe in self.pipes if pipe.x + PIPE_WIDTH > 0]

            # Identify closest pipe for decision-making
            closest_pipe = None
            for pipe in self.pipes:
                # As soon as we find one that hasn't gone past the bird's x, break
                if pipe.x + PIPE_WIDTH > self.birds[0].x:
                    closest_pipe = pipe
                    break

            # Update each bird
            alive_count = 0
            for bird in self.birds:
                if not bird.alive:
                    continue

                bird.decide(closest_pipe)
                bird.update()
                bird_rect = bird.get_rect()

                # Boundary collision
                if bird.y < 0 or bird.y + BIRD_HEIGHT > HEIGHT:
                    bird.alive = False

                # Pipe collision
                for pipe in self.pipes:
                    if (bird_rect.colliderect(pipe.get_top_rect()) or
                            bird_rect.colliderect(pipe.get_bottom_rect())):
                        bird.alive = False

                # Passing pipes (score + fitness bonus)
                for pipe in self.pipes:
                    if (pipe.id not in bird.passed_pipes) and (bird.x > pipe.x + PIPE_WIDTH):
                        bird.score += 1
                        bird.fitness += 100
                        bird.passed_pipes.add(pipe.id)

                if bird.alive:
                    alive_count += 1

            # Track best fitness for lifetime display
            current_score = max(b.score for b in self.birds)
            current_gen_best = max(b.fitness for b in self.birds)
            if current_gen_best > self.best_lifetime_fitness:
                self.best_lifetime_fitness = current_gen_best

            if current_score >= self.last_checkpoint_pipe + 5000:
                if current_gen_best > self.checkpoint_fitness:
                    best_bird = max(self.birds, key=lambda b: b.fitness)
                    save_checkpoint(best_bird.brain, best_bird.fitness, CHECKPOINT_FILE)
                    self.checkpoint_fitness = best_bird.fitness
                    self.last_checkpoint_pipe = current_score
                    print(f"Mid-generation checkpoint saved at score {current_score}!")

            # If all birds are dead, prepare for the next generation.
            if alive_count == 0:
                # Always update the checkpoint if the current generation's best fitness is improved.
                if current_gen_best > self.checkpoint_fitness:
                    best_bird = max(self.birds, key=lambda b: b.fitness)
                    save_checkpoint(best_bird.brain, best_bird.fitness, CHECKPOINT_FILE)
                    self.checkpoint_fitness = best_bird.fitness

                self.generation += 1
                self.birds = next_generation(self.birds, self.population_size)
                self.pipes = [Pipe()]
                self.frame_count = 0

            # Draw everything
            self.screen.fill(BLUE)
            for pipe in self.pipes:
                pygame.draw.rect(self.screen, GREEN, pipe.get_top_rect())
                pygame.draw.rect(self.screen, GREEN, pipe.get_bottom_rect())
            for bird in self.birds:
                if bird.alive:
                    pygame.draw.rect(self.screen, BLACK, bird.get_rect())

            self.draw_scoreboard()
            pygame.display.flip()

def main():
    parser = argparse.ArgumentParser(description="Flappy Bird (Genetic Algorithm Mode)")
    parser.add_argument("--no-checkpoint", dest="use_checkpoint", action="store_false",
                        help="If set, do not load the checkpoint weights into the population. (Default: load checkpoint)")
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE,
                        help="Population size for the GA (default: 100).")
    parser.add_argument("--injection_rate", type=float, default=0.2,
                        help="Fraction of new population that receives the checkpoint network if loaded.")
    parser.set_defaults(use_checkpoint=True)
    args = parser.parse_args()

    game = FlappyGAGame(
        use_checkpoint=args.use_checkpoint, 
        population_size=args.population_size, 
        injection_rate=args.injection_rate
    )
    game.run()

if __name__ == "__main__":
    main()