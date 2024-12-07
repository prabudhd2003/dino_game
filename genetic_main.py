import pygame
import random
import numpy as np

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
GROUND_HEIGHT = 350
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Dino:
    def __init__(self):
        self.dino_rect = pygame.Rect(50, GROUND_HEIGHT, 40, 40)
        self.gravity = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.gravity = -10
            self.is_jumping = True

    def update(self):
        self.gravity += 0.5
        self.dino_rect.y += self.gravity
        if self.dino_rect.y >= GROUND_HEIGHT:
            self.dino_rect.y = GROUND_HEIGHT
            self.is_jumping = False

    def draw(self, screen):
        pygame.draw.rect(screen, (0, 255, 0), self.dino_rect)

class Obstacle:
    def __init__(self, x=None):
        self.obstacle_rect = pygame.Rect(x or SCREEN_WIDTH, SCREEN_HEIGHT - 40, 20, 40)
        self.speed = 5

    def update(self, speed):
        self.obstacle_rect.x -= speed

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.obstacle_rect)

class ThickObstacle(Obstacle):
    def __init__(self, x=None):
        super().__init__(x)
        self.obstacle_rect.width *= 1.5  # Make the obstacle thicker
        self.obstacle_rect.height *= 1.5  # Optionally make it taller too

    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.obstacle_rect)

def game_loop(genomes, generation, best_generation, highest_score):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    dinos = [Dino() for _ in genomes]
    obstacles = []
    scores = [0 for _ in genomes]

    base_speed = 5
    speed_increment = 0.01  # Speed increases as score increases
    current_speed = base_speed

    # Timing and spacing variables
    obstacle_timer = 0
    min_obstacle_interval = 1500  # Minimum time (in ms) between obstacle batches
    max_obstacle_interval = 2500  # Maximum time (in ms) between obstacle batches
    min_obstacle_distance = 150  # Minimum distance between obstacles
    batch_chance_increase = 0.01  # Increase chance of 2-obstacle batch as speed increases

    running = True
    while running:
        screen.fill(WHITE)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for i, dino in enumerate(dinos):
            if not genomes[i]['alive']:
                continue

            if obstacles:
                inputs = np.array([
                    obstacles[0].obstacle_rect.x - dino.dino_rect.x,
                    dino.dino_rect.y,
                    obstacles[0].obstacle_rect.y,
                ])
            else:
                inputs = np.array([0, dino.dino_rect.y, 0])

            output = genomes[i]['nn'].predict(inputs)

            if output > 0.5:
                dino.jump()

            dino.update()

            if obstacles and dino.dino_rect.colliderect(obstacles[0].obstacle_rect):
                genomes[i]['alive'] = False  # Mark the player as dead

            else:
                scores[i] += 1  # Incrementing the score by whole numbers

            dino.draw(screen)

        # Adjust game speed based on the highest score
        current_speed = base_speed + max(scores) * speed_increment

        # Update obstacle timer
        obstacle_timer += clock.get_time()

        # Spawn obstacles in batches and ensure spacing
        if len(obstacles) == 0 or (obstacles[-1].obstacle_rect.x < SCREEN_WIDTH - min_obstacle_distance and obstacle_timer > min_obstacle_interval):
            obstacle_timer = 0

            # Determine if we generate a single obstacle or a batch
            batch_chance = min(0.5 + (current_speed - base_speed) * batch_chance_increase, 0.9)
            num_obstacles = 2 if random.random() < batch_chance else 1

            for _ in range(num_obstacles):
                if random.random() < 0.5:
                    obstacles.append(Obstacle())
                else:
                    obstacles.append(ThickObstacle())

                if num_obstacles == 2:
                    obstacles[-1].obstacle_rect.x += min_obstacle_distance  # Space the obstacles apart

        for obstacle in obstacles:
            obstacle.update(current_speed)
            obstacle.draw(screen)

        # Remove obstacles that have moved off the screen
        obstacles = [obstacle for obstacle in obstacles if obstacle.obstacle_rect.right > 0]

        # Count how many players are still alive
        alive_count = sum(genome['alive'] for genome in genomes)

        # Display generation info
        generation_text = font.render(f"Generation: {generation}", True, BLACK)
        highest_score_text = font.render(f"Highest Score: {highest_score} (Gen {best_generation})", True, BLACK)
        current_score_text = font.render(f"Current High Score: {max(scores)}", True, BLACK)  # Updated for whole numbers
        alive_text = font.render(f"Players Alive: {alive_count}", True, BLACK)

        screen.blit(generation_text, (10, 10))
        screen.blit(highest_score_text, (10, 50))
        screen.blit(current_score_text, (10, 90))
        screen.blit(alive_text, (10, 130))

        pygame.display.update()
        clock.tick(30 + int(current_speed))  # Increasing speed over time

        # End the game loop if all dinos are dead
        if alive_count == 0:
            running = False

    pygame.quit()

    return scores

class NeuralNetwork:
    def __init__(self):
        # Simple neural network with random weights for demonstration
        self.weights = np.random.randn(3)

    def predict(self, inputs):
        return np.dot(inputs, self.weights)  # Return the scalar result directly

def crossover(parent1, parent2):
    child_weights = np.where(np.random.rand(3) < 0.5, parent1.weights, parent2.weights)
    child = NeuralNetwork()
    child.weights = child_weights
    return child

def mutate(nn):
    mutation_strength = 0.1
    nn.weights += np.random.randn(3) * mutation_strength

def genetic_algorithm():
    population_size = 10
    generations = 100
    mutation_rate = 0.001

    population = [{'nn': NeuralNetwork(), 'alive': True} for _ in range(population_size)]

    highest_score = 0
    best_generation = 0
    generation_scores = []

    for generation in range(generations):
        print(f'Generation {generation+1}')

        scores = game_loop(population, generation+1, best_generation, highest_score)
        max_score = max(scores)
        generation_scores.append(max_score)

        if max_score > highest_score:
            highest_score = max_score
            best_generation = generation + 1

        best_genomes = [population[i] for i in np.argsort(scores)[-5:]]

        new_population = []
        for _ in range(population_size):
            parent1, parent2 = random.sample(best_genomes, 2)
            child = crossover(parent1['nn'], parent2['nn'])

            if random.random() < mutation_rate:
                mutate(child)

            new_population.append({'nn': child, 'alive': True})

        population = new_population

        print(f"Highest Score so far: {highest_score} (Gen {best_generation})")

if __name__ == "__main__":
    genetic_algorithm()
