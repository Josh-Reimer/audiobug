import time
import audioop
import pygame
import pyaudio
import math

# Initialisation for PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# PyGame initialisations and basic objects
pygame.init()
screensize = (900, 600)
screen = pygame.display.set_mode(screensize)
pygame.display.set_caption("Shout harder.. :D")

# Defining colors
WHITE = (255, 255, 255)
RED = (255, 128, 128)
YELLOW = (255, 255, 128)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)

# Font setup
font = pygame.font.SysFont('Arial', 24)
title_font = pygame.font.SysFont('Arial', 32, bold=True)

# Variables
margin = 20
samples_per_section = screensize[0] // 3 - 2 * margin  # Integer division

# Initialize sound tracks and max values
sound_tracks = [[0] * samples_per_section for _ in range(3)]
max_value = [0] * 3
current_section = 0

# Decibel tracking
current_db = 0
max_db = [-120] * 3  # Initialize with very low values

# Loop till close button clicked
done = False
clock = pygame.time.Clock()

def calculate_decibel(amplitude):
    """Convert amplitude to decibels"""
    if amplitude <= 0:
        return -120  # Silence
    
    # Calculate dB relative to maximum possible amplitude (32767 for 16-bit signed)
    try:
        db = 20 * math.log10(amplitude / 32767)
    except ValueError:
        return -120
    
    # Scale to a more meaningful range (0 = loudest, -120 = quietest)
    return max(db, -120)

while not done:
    total = 0
    # Read data from device for around 0.2 seconds
    for _ in range(0, 20):  # Increased samples for better responsiveness
        data = stream.read(CHUNK, exception_on_overflow=False)
        reading = audioop.max(data, 2)
        total += reading
        time.sleep(0.0001)

    # Scale the reading
    total = total // 100  # Integer division
    
    # Calculate decibel level
    current_db = calculate_decibel(total)

    # Update current section data
    sound_tracks[current_section] = sound_tracks[current_section][1:] + [total]
    max_value[current_section] = max(max_value[current_section], total)
    
    # Update max dB for current section
    section_db = calculate_decibel(max_value[current_section])
    if section_db > max_db[current_section]:
        max_db[current_section] = section_db

    screen.fill(WHITE)

    # Draw title
    title = title_font.render("Sound Level Monitor", True, BLACK)
    screen.blit(title, (screensize[0] // 2 - title.get_width() // 2, 10))

    # Draw highlighted section
    section_width = screensize[0] // 3
    pygame.draw.rect(screen, YELLOW, (section_width * current_section, 50, section_width, screensize[1] - 50))

    # Draw all three sections
    for i in range(3):
        section_x = i * section_width + margin
        
        # Draw max value bar
        bar_height = min(max_value[i], screensize[1] - 100)  # Prevent overflow
        pygame.draw.rect(screen, RED, (section_x, screensize[1] - bar_height, 
                                       section_width - 2 * margin, bar_height))
        
        # Draw waveform
        for j in range(samples_per_section):
            x = section_x + j
            value = min(sound_tracks[i][j], screensize[1] - 100)  # Prevent overflow
            y = screensize[1] - value
            pygame.draw.rect(screen, BLUE, (x, y, 1, value))
        
        # Draw decibel info for each section
        section_title = font.render(f"Section {i+1}", True, BLACK)
        screen.blit(section_title, (i * section_width + section_width // 2 - section_title.get_width() // 2, 60))
        
        # Current max for section
        max_db_text = font.render(f"Max: {max_db[i]:.1f} dB", True, BLACK)
        screen.blit(max_db_text, (i * section_width + section_width // 2 - max_db_text.get_width() // 2, 90))
        
        # Highlight current section with current dB
        if i == current_section:
            db_text = font.render(f"Current: {current_db:.1f} dB", True, GREEN)
            screen.blit(db_text, (i * section_width + section_width // 2 - db_text.get_width() // 2, 120))

    # Draw overall info at top
    overall_info = font.render(f"Current dB: {current_db:.1f} | Max dB: {max_db[current_section]:.1f}", True, BLACK)
    screen.blit(overall_info, (screensize[0] // 2 - overall_info.get_width() // 2, screensize[1] - 40))

    # Draw instructions at bottom
    instructions = font.render("Left-click: Select section | Right-click: Reset all", True, BLACK)
    screen.blit(instructions, (screensize[0] // 2 - instructions.get_width() // 2, screensize[1] - 70))

    # Update display
    pygame.display.flip()

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:  # Right click
                # Reset all sections
                sound_tracks = [[0] * samples_per_section for _ in range(3)]
                max_value = [0] * 3
                max_db = [-120] * 3
            else:  # Left click
                pos = pygame.mouse.get_pos()
                current_section = min(pos[0] * 3 // screensize[0], 2)  # Integer division
                print(f"Clicked at {pos}, section: {current_section}")

# Clean up resources
stream.stop_stream()
stream.close()
p.terminate()
pygame.quit()