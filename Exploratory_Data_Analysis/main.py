import pygame
import random
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import time
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Initialize pygame
pygame.init()

# Set screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Set colors
white = (255, 255, 255)
black = (0, 0, 0)
red = (255, 0, 0)
green = (0, 255, 0)
grey = (169, 169, 169)
blue = (0, 0, 255)

# Fonts
font = pygame.font.SysFont(None, 55)
small_font = pygame.font.SysFont(None, 30)

# Game Variables
clock = pygame.time.Clock()
car_width, car_height = 50, 50
obstacle_width, obstacle_height = 50, 50
speed = 5
training_data = []
csv_file = 'training_data.csv'

# Initialize metrics
training_start_time = None
successful_decisions = 0
total_decisions = 0
agent_count = 0

# Load car image
car_img = pygame.Surface((car_width, car_height))
car_img.fill(green)

# Load obstacle image
obstacle_img = pygame.Surface((obstacle_width, obstacle_height))
obstacle_img.fill(red)

# Display Text
def display_text(text, x, y, font, color=black):
    screen_text = font.render(text, True, color)
    screen.blit(screen_text, [x, y])

# Draw road
def draw_road():
    road_width = 400
    road_x = (screen_width - road_width) // 2
    pygame.draw.rect(screen, grey, (road_x, 0, road_width, screen_height))

# Main Menu
def main_menu():
    menu = True
    while menu:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(white)
        display_text("Main Menu", 300, 100, font)
        display_text("Train", 350, 250, font)
        display_text("Test", 350, 350, font)

        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if 350 < mouse[0] < 450 and 250 < mouse[1] < 300:
            if click[0] == 1:
                train()

        if 350 < mouse[0] < 450 and 350 < mouse[1] < 400:
            if click[0] == 1:
                test()

        pygame.display.update()
        clock.tick(15)

# Train Scene
def train():
    global training_data, training_start_time, successful_decisions, total_decisions, agent_count
    training = True
    car_x, car_y = screen_width // 2, screen_height - 100

    # Load existing data if available
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        training_data = df.values.tolist()
    else:
        training_data = []

    # Initialize RandomForestClassifier
    if training_data:
        df = pd.DataFrame(training_data, columns=['car_x', 'car_y', 'obstacle_x', 'obstacle_y', 'reward'])
        X = df[['car_x', 'obstacle_x', 'obstacle_y']]
        y = df['car_y']
        model = RandomForestClassifier()
        model.fit(X, y)
    else:
        model = None

    training_start_time = time.time()
    while training:
        car_y_pred = None  # Initialize car_y_pred here

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    training = False
                    main_menu()

        draw_road()
        screen.blit(car_img, (car_x, car_y))
        obstacle_x = random.randint(200, screen_width - 200 - obstacle_width)
        obstacle_y = -50
        agent_count += 1

        while True:
            screen.fill(white)
            draw_road()
            screen.blit(car_img, (car_x, car_y))
            screen.blit(obstacle_img, (obstacle_x, obstacle_y))
            obstacle_y += speed

            if obstacle_y > screen_height:
                obstacle_y = -50
                obstacle_x = random.randint(200, screen_width - 200 - obstacle_width)
                agent_count += 1

            car_movement = [0, 0]

            if model:
                car_y_pred = model.predict([[car_x, obstacle_x, obstacle_y]])[0]
                if car_y_pred < car_y:
                    car_movement[1] = -1
                elif car_y_pred > car_y:
                    car_movement[1] = 1

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                car_movement[0] = -1
            if keys[pygame.K_RIGHT]:
                car_movement[0] = 1

            car_x += car_movement[0] * speed
            car_y += car_movement[1] * speed

            if car_x < 200:
                car_x = 200
            elif car_x > screen_width - 200 - car_width:
                car_x = screen_width - 200 - car_width

            reward = 1 if car_y_pred is not None and car_y == car_y_pred else 0
            training_data.append([car_x, car_y, obstacle_x, obstacle_y, reward])

            total_decisions += 1
            successful_decisions += reward

            # Update model with new data
            if len(training_data) > 100:
                df = pd.DataFrame(training_data, columns=['car_x', 'car_y', 'obstacle_x', 'obstacle_y', 'reward'])
                X = df[['car_x', 'obstacle_x', 'obstacle_y']]
                y = df['car_y']
                model = RandomForestClassifier()
                model.fit(X, y)

            # Save training data
            df = pd.DataFrame(training_data, columns=['car_x', 'car_y', 'obstacle_x', 'obstacle_y', 'reward'])
            df.to_csv(csv_file, index=False)

            # Display metrics
            elapsed_time = int(time.time() - training_start_time)
            prediction_rate = (successful_decisions / total_decisions) * 100 if total_decisions else 0
            display_text(f"Time: {elapsed_time}s", 650, 10, small_font)
            display_text(f"Agents: {agent_count}", 650, 40, small_font)
            display_text(f"Success: {successful_decisions}", 650, 70, small_font)
            display_text(f"Accuracy: {prediction_rate:.2f}%", 650, 100, small_font)
            display_text("Back", 10, 10, small_font)

            pygame.display.update()
            clock.tick(30)

            if not training:
                break

# Test Scene
def test():
    testing = True
    df = pd.read_csv(csv_file)
    X = df[['car_x', 'obstacle_x', 'obstacle_y']]
    y = df['car_y']
    model = RandomForestClassifier()
    model.fit(X, y)

    while testing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    testing = False
                    main_menu()

        screen.fill(white)
        display_text("Test", 350, 100, font)
        display_text("Begin", 350, 250, font)
        display_text("Back", 10, 10, small_font)

        mouse = pygame.mouse.get_pos()
        click = pygame.mouse.get_pressed()

        if 350 < mouse[0] < 450 and 250 < mouse[1] < 300:
            if click[0] == 1:
                run_test(model)

        pygame.display.update()
        clock.tick(15)

# Run Test
def run_test(model):
    running = True
    car_x, car_y = screen_width // 2, screen_height - 100
    obstacle_x = random.randint(200, screen_width - 200 - obstacle_width)
    obstacle_y = -50

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.fill(white)
        draw_road()
        screen.blit(car_img, (car_x, car_y))
        screen.blit(obstacle_img, (obstacle_x, obstacle_y))
        obstacle_y += speed

        if obstacle_y > screen_height:
            obstacle_y = -50
            obstacle_x = random.randint(200, screen_width - 200 - obstacle_width)

        car_y_pred = model.predict([[car_x, obstacle_x, obstacle_y]])[0]
        car_y = int(car_y_pred)

        car_x += random.choice([-1, 1]) * speed

        if car_x < 200:
            car_x = 200
        elif car_x > screen_width - 200 - car_width:
            car_x = screen_width - 200 - car_width

        pygame.display.update()
        clock.tick(30)
        screen.fill(white)
        draw_road()
        screen.blit(car_img, (car_x, car_y))
        screen.blit(obstacle_img, (obstacle_x, obstacle_y))
        obstacle_y += speed

        if obstacle_y > screen_height:
            obstacle_y = -50
            obstacle_x = random.randint(200, screen_width - 200 - obstacle_width)

        car_y_pred = model.predict([[car_x, obstacle_x, obstacle_y]])[0]
        car_y = int(car_y_pred)

        car_x += random.choice([-1, 1]) * speed

        if car_x < 200:
            car_x = 200
        elif car_x > screen_width - 200 - car_width:
            car_x = screen_width - 200 - car_width

        # Check collision
        if (car_x < obstacle_x < car_x + car_width or car_x < obstacle_x + obstacle_width < car_x + car_width) and \
                (car_y < obstacle_y < car_y + car_height or car_y < obstacle_y + obstacle_height < car_y + car_height):
            display_text("Collision!", 350, 300, font, red)
            running = False

        pygame.display.update()
        clock.tick(30)

# Start the game
main_menu()
