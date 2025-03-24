import cv2
import mediapipe as mp
import pygame
import sys
import random
import math
import time  # For tongue-out delay timing

# ================================================================
# MediaPipe Setup
# ================================================================
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# ================================================================
# Global Constants (Game)
# ================================================================
WIDTH, HEIGHT = 800, 600
FPS = 30

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

PADDLE_WIDTH, PADDLE_HEIGHT = 20, 100
BALL_SIZE = 20
BALL_SPEED = 10

LEVEL_INCREMENT=6
MENU_OPTIONS = ["Continue", "Restart", "Switch Hand"]
MENU_OPTION_HEIGHT = 50
MENU_MARGIN = 20

# Game States
STATE_HAND_SELECTION = 0
STATE_PLAYING = 1
STATE_PAUSED = 2
STATE_GAME_OVER = 3

TONGUE_DELAY = 1.0

# ---------------------------------------------------------
# PADDING / MARGIN SETTINGS for hand-to-paddle mapping
# ---------------------------------------------------------
# Adjust these so that even if your hand is not exactly at the edge,
# the paddle still reaches near the top and bottom.
HAND_TOP_PADDING = -0.5  
HAND_BOTTOM_PADDING = -0.25

# ================================================================
# Gesture Detection Helpers
# ================================================================
def measure_fistness(landmarks):
    fingertip_indices = [4, 8, 12, 16, 20]
    wrist = landmarks[0]
    total_dist = 0.0
    for idx in fingertip_indices:
        tip = landmarks[idx]
        dist = math.dist((wrist.x, wrist.y), (tip.x, tip.y))
        total_dist += dist
    return total_dist

def hand_label_to_side(classification):
    return classification.label.lower()

def detect_tongue_out(face_mesh_results, open_mouth_threshold=0.05):
    if not face_mesh_results or not face_mesh_results.multi_face_landmarks:
        return False
    face_landmarks = face_mesh_results.multi_face_landmarks[0].landmark
    top_lip = face_landmarks[13]
    bottom_lip = face_landmarks[14]
    dist = math.dist((top_lip.x, top_lip.y), (bottom_lip.x, bottom_lip.y))
    return dist > open_mouth_threshold

def get_mediapipe_data(cap, hands, face_mesh):
    success, frame = cap.read()
    if not success:
        return 0.0, 0.0, False, False, False, None

    # Flip horizontally so your physical left is left on screen
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb) if face_mesh else None

    left_y, right_y = 0.0, 0.0
    left_fist, right_fist = False, False

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for hand_landmarks, hand_classification in zip(
            results_hands.multi_hand_landmarks,
            results_hands.multi_handedness
        ):
            side = hand_label_to_side(hand_classification.classification[0])
            lm = hand_landmarks.landmark
            fistness = measure_fistness(lm)
            is_fist = (fistness < 1.0)
            wrist_y = lm[0].y  # normalized [0..1]
            if side == 'left':
                left_y = wrist_y
                left_fist = is_fist
            else:
                right_y = wrist_y
                right_fist = is_fist

    tongue_out = False
    if results_face and results_face.multi_face_landmarks:
        tongue_out = detect_tongue_out(results_face)
    return left_y, right_y, left_fist, right_fist, tongue_out, frame

# ================================================================
# Pygame Rendering Functions
# ================================================================
def render_score_and_lives(screen, score, lives, level):
    font = pygame.font.SysFont(None, 36)
    text_surface = font.render(f"Score: {score}   Lives: {lives}   Level: {level}", True, WHITE)
    screen.blit(text_surface, (10, 10))

def render_game_paused(screen, selected_option_idx, score, lives):
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 48)
    pause_text = font.render("PAUSED", True, WHITE)
    screen.blit(pause_text, (WIDTH // 2 - 80, HEIGHT // 4))
    info_text = font.render(f"Score: {score}  Lives: {lives}", True, WHITE)
    screen.blit(info_text, (WIDTH // 2 - 150, HEIGHT // 4 + 60))
    for i, option in enumerate(MENU_OPTIONS):
        color = WHITE
        if i == selected_option_idx:
            rendered_option = font.render(f"> {option} <", True, color)
        else:
            rendered_option = font.render(option, True, color)
        screen.blit(
            rendered_option,
            (WIDTH // 2 - rendered_option.get_width() // 2,
             HEIGHT // 2 + i * 60)
        )

def render_game_over(screen, score):
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 60)
    game_over_text = font.render("GAME OVER", True, WHITE)
    screen.blit(game_over_text, (WIDTH // 2 - 140, HEIGHT // 3))
    font_small = pygame.font.SysFont(None, 36)
    score_text = font_small.render(f"Final Score: {score}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - 80, HEIGHT // 3 + 60))
    info_text = font_small.render("Stick out your tongue to restart!", True, WHITE)
    screen.blit(info_text, (WIDTH // 2 - 180, HEIGHT // 3 + 120))

def render_hand_selection(screen):
    screen.fill(BLACK)
    font = pygame.font.SysFont(None, 48)
    text = font.render("Raise the hand you'd like to play with!", True, WHITE)
    screen.blit(text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - 40))

def display_webcam_in_corner(screen, frame):
    if frame is None:
        return
    preview_width = 160
    preview_height = 120
    frame_small = cv2.resize(frame, (preview_width, preview_height),
                             interpolation=cv2.INTER_AREA)
    frame_small_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    webcam_surf = pygame.image.frombuffer(frame_small_rgb.tobytes(), (preview_width, preview_height), "RGB")
    x_pos = WIDTH - preview_width - 10
    y_pos = 10
    screen.blit(webcam_surf, (x_pos, y_pos))


# ================================================================
# Global Game Variables (for pause selection)
# ================================================================
game_state = STATE_HAND_SELECTION
lives = 3
score = 0
control_hand = None

def handle_pause_selection(selected_option_idx):
    global game_state, lives, score, control_hand
    option = MENU_OPTIONS[selected_option_idx]
    if option == "Continue":
        game_state = STATE_PLAYING
    elif option == "Restart":
        lives = 3
        score = 0
        control_hand = None
        game_state = STATE_HAND_SELECTION
    elif option == "Switch Hand":
        # Swap the current control hand
        if control_hand == 'left':
            control_hand = 'right'
        else:
            control_hand = 'left'
        game_state = STATE_PLAYING

# ================================================================
# Main Game Loop
# ================================================================
def main():
    global game_state, lives, score, control_hand
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Motion-Controlled Pong (MediaPipe)")
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access webcam.")
        sys.exit(1)

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands, mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        # Paddle
        paddle_x = 50
        paddle_y = HEIGHT // 2 - PADDLE_HEIGHT // 2

        # Ball
        ball_x = WIDTH // 2
        ball_y = HEIGHT // 2
        ball_vel_x = BALL_SPEED * random.choice([-1, 1])
        ball_vel_y = BALL_SPEED * random.choice([-1, 1])

        # Track the number of paddle hits (returns) for speed increase and level
        bounce_count = 0

        selected_option_idx = 0
        last_tongue_time = 0.0

        def reset_ball():
            nonlocal ball_x, ball_y, ball_vel_x, ball_vel_y
            ball_x = WIDTH // 2
            ball_y = HEIGHT // 2
            # Reset speed to base BALL_SPEED
            #ball_vel_x = BALL_SPEED * random.choice([-1, 1])
            #ball_vel_y = BALL_SPEED * random.choice([-1, 1])

        def reset_game():
            global lives, score, game_state, control_hand
            lives = 3
            score = 0
            control_hand = None
            game_state = STATE_HAND_SELECTION
            nonlocal ball_vel_x, ball_vel_y, bounce_count
            ball_vel_x = BALL_SPEED * random.choice([-1, 1])
            ball_vel_y = BALL_SPEED * random.choice([-1, 1])
            bounce_count = 0

        running = True
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            left_y, right_y, left_fist, right_fist, tongue_out, frame = get_mediapipe_data(cap, hands, face_mesh)
            now = time.time()

            # -------------------------------------------------------------------
            # HAND SELECTION PHASE
            # -------------------------------------------------------------------
            if game_state == STATE_HAND_SELECTION:
                render_hand_selection(screen)
                display_webcam_in_corner(screen, frame)
                pygame.display.flip()
                left_raised = (left_y < 0.5)
                right_raised = (right_y < 0.5)
                if left_raised ^ right_raised:
                    # If it's reversed for you, swap these assignments.
                    if left_raised:
                        control_hand = 'right'
                    else:
                        control_hand = 'left'
                    game_state = STATE_PLAYING
                continue

            # -------------------------------------------------------------------
            # GAME OVER PHASE
            # -------------------------------------------------------------------
            if game_state == STATE_GAME_OVER:
                if tongue_out and (now - last_tongue_time) > TONGUE_DELAY:
                    ball_vel_x = BALL_SPEED * random.choice([-1, 1])
                    ball_vel_y = BALL_SPEED * random.choice([-1, 1])
                    bounce_count = 0
                    lives = 3
                    score = 0
                    control_hand = None
                    game_state = STATE_HAND_SELECTION
                    last_tongue_time = now
                render_game_over(screen, score)
                display_webcam_in_corner(screen, frame)
                pygame.display.flip()
                continue

            # -------------------------------------------------------------------
            # PAUSED PHASE
            # -------------------------------------------------------------------
            if game_state == STATE_PAUSED:
                if control_hand == 'left':
                    hand_y = left_y
                else:
                    hand_y = right_y
                if hand_y < 0.33:
                    selected_option_idx = 0
                elif hand_y < 0.66:
                    selected_option_idx = 1
                else:
                    selected_option_idx = 2
                if tongue_out and (now - last_tongue_time) > TONGUE_DELAY:
                    handle_pause_selection(selected_option_idx)
                    last_tongue_time = now
                render_game_paused(screen, selected_option_idx, score, lives)
                display_webcam_in_corner(screen, frame)
                pygame.display.flip()
                continue

            # -------------------------------------------------------------------
            # PLAYING PHASE
            # -------------------------------------------------------------------
            if game_state == STATE_PLAYING:
                if tongue_out and (now - last_tongue_time) > TONGUE_DELAY:
                    game_state = STATE_PAUSED
                    last_tongue_time = now

                # -------------------------
                # PADDLE Movement with Padding
                # -------------------------
                if control_hand == 'left':
                    hand_val = left_y
                else:
                    hand_val = right_y
                hand_val = max(0.0, min(1.0, hand_val))
                range_y = 1.0 - (HAND_TOP_PADDING + HAND_BOTTOM_PADDING)
                mapped = HAND_TOP_PADDING + hand_val * range_y
                paddle_y = int(mapped * (HEIGHT - PADDLE_HEIGHT))

                # Move ball
                ball_x += ball_vel_x
                ball_y += ball_vel_y

                if ball_y <= 0 or (ball_y + BALL_SIZE >= HEIGHT):
                    ball_vel_y = -ball_vel_y

                # Paddle collision
                if (paddle_x < ball_x < paddle_x + PADDLE_WIDTH) and (paddle_y < ball_y + BALL_SIZE < paddle_y + PADDLE_HEIGHT):
                    ball_vel_x = -ball_vel_x
                    score += 1
                    bounce_count += 1
                    # Every 6 bounces, increase ball speed by a factor of 1.25 and record message time.
                    if bounce_count % LEVEL_INCREMENT == 0:
                        ball_vel_x *= 1.25
                        ball_vel_y *= 1.25

                if ball_x + BALL_SIZE >= WIDTH:
                    ball_vel_x = -ball_vel_x

                if ball_x <= 0:
                    lives -= 1
                    if lives <= 0:
                        game_state = STATE_GAME_OVER
                    else:
                        reset_ball()

                screen.fill(BLACK)
                pygame.draw.rect(screen, WHITE, (paddle_x, paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
                pygame.draw.rect(screen, WHITE, (ball_x, ball_y, BALL_SIZE, BALL_SIZE))
                # Compute level as bounce_count // 6 + 1
                level = bounce_count // LEVEL_INCREMENT + 1
                render_score_and_lives(screen, score, lives, level)

                display_webcam_in_corner(screen, frame)
                pygame.display.flip()
                continue

        cap.release()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
