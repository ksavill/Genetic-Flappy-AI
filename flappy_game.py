import pygame
import random
import sys

# ----------------------- Global Configuration -----------------------
WIDTH, HEIGHT = 400, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (135, 206, 235)

FPS = 60
PIPE_WIDTH = 70
PIPE_GAP = 150
PIPE_VELOCITY = 3

BIRD_WIDTH, BIRD_HEIGHT = 30, 30
GRAVITY = 0.5
JUMP_STRENGTH = -10

__all__ = [
    "WIDTH",
    "HEIGHT",
    "WHITE",
    "BLACK",
    "GREEN",
    "BLUE",
    "FPS",
    "PIPE_WIDTH",
    "PIPE_GAP",
    "PIPE_VELOCITY",
    "BIRD_WIDTH",
    "BIRD_HEIGHT",
    "GRAVITY",
    "JUMP_STRENGTH",
    "create_pipe",
    "FlappyGame"
]

def create_pipe():
    """Create a new pair of pipes with a gap in between."""
    pipe_top_height = random.randint(50, HEIGHT - PIPE_GAP - 50)
    top_pipe = pygame.Rect(WIDTH, 0, PIPE_WIDTH, pipe_top_height)
    bottom_pipe = pygame.Rect(
        WIDTH,
        pipe_top_height + PIPE_GAP,
        PIPE_WIDTH,
        HEIGHT - (pipe_top_height + PIPE_GAP)
    )
    return [top_pipe, bottom_pipe, False]

class FlappyGame:
    """
    Encapsulates the manual (single-bird) Flappy Bird game.
    """

    def __init__(self, caption="Flappy Bird (User Mode)"):
        pygame.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        # ------------------- Font size changed to 24 -------------------
        self.font = pygame.font.SysFont(None, 24)

        # Bird state
        self.bird_x = WIDTH // 4
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0

        # Pipe tracking
        self.pipes = []   # list of [top_rect, bottom_rect, scored_flag]
        self.score = 0

        # Timing
        self.pipe_frequency = 1500  # in milliseconds
        self.last_pipe_time = pygame.time.get_ticks()

        self.game_over = False

    def reset_game(self):
        """Reset key variables for a new manual game."""
        self.bird_y = HEIGHT // 2
        self.bird_velocity = 0
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.last_pipe_time = pygame.time.get_ticks()

    def draw_scoreboard(self):
        """Draw the standard scoreboard at the top-left."""
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

    def run(self):
        """Main game loop for manual play."""
        while True:
            self.clock.tick(FPS)
            # ------------------- Handle Events -------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not self.game_over:
                        self.bird_velocity = JUMP_STRENGTH
                    elif event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            # ------------------- Game Logic ----------------------
            if not self.game_over:
                # Update bird physics
                self.bird_velocity += GRAVITY
                self.bird_y += self.bird_velocity
                bird_rect = pygame.Rect(self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)

                # Spawn new pipes
                current_time = pygame.time.get_ticks()
                if current_time - self.last_pipe_time > self.pipe_frequency:
                    self.last_pipe_time = current_time
                    self.pipes.append(create_pipe())

                # Move pipes, check collisions, update score
                for pipe in self.pipes:
                    top_pipe, bottom_pipe, scored_flag = pipe
                    top_pipe.x -= PIPE_VELOCITY
                    bottom_pipe.x -= PIPE_VELOCITY

                    if not scored_flag and top_pipe.x + PIPE_WIDTH < self.bird_x:
                        self.score += 1
                        pipe[2] = True

                    if bird_rect.colliderect(top_pipe) or bird_rect.colliderect(bottom_pipe):
                        self.game_over = True

                # Remove off-screen pipes
                self.pipes = [
                    pipe for pipe in self.pipes
                    if pipe[0].x + PIPE_WIDTH > 0
                ]

                # Check if bird out of screen bounds
                if self.bird_y < 0 or self.bird_y + BIRD_HEIGHT > HEIGHT:
                    self.game_over = True

            # ------------------- Drawing -------------------------
            self.screen.fill(BLUE)

            # Bird
            pygame.draw.rect(self.screen, BLACK, (self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))

            # Pipes
            for pipe in self.pipes:
                top_pipe, bottom_pipe, _ = pipe
                pygame.draw.rect(self.screen, GREEN, top_pipe)
                pygame.draw.rect(self.screen, GREEN, bottom_pipe)

            # Scoreboard
            self.draw_scoreboard()

            # Game-over message
            if self.game_over:
                msg = "Game Over! Press R to Restart"
                text_surf = self.font.render(msg, True, WHITE)
                self.screen.blit(
                    text_surf,
                    (
                        WIDTH // 2 - text_surf.get_width() // 2,
                        HEIGHT // 2 - text_surf.get_height() // 2
                    )
                )

            pygame.display.flip()

def main():
    game = FlappyGame()  # uses default title "Flappy Bird (Manual)"
    game.run()

if __name__ == "__main__":
    main()
