from PIL import Image, ImageFilter
import time
import re

from time import sleep

if __name__ == '__main__':

    timestamp = 0
    last_timestamp = 0
    file_ready = False
    mario_position = []
    enemies = []

    while True:
        with open("extracted_data.txt", "r") as f:
            lines = f.readlines()

            enemies = []

            for i in range(0, len(lines)):
                line = lines[i].strip()
                tokens = line.split(":")

                if tokens[0] == 'MarioPosition':
                    mario_position = tokens[1].split(",")
                elif tokens[0].startswith('Enemy'):
                    enemies.append(tokens[1].split(","))
                elif tokens[0] == 'Timestamp':
                    timestamp = tokens[1]
                    file_ready = True

        # Timestamp is the last line in the file. If it is not found, the data is not complete in the file. 
        # If the timestamp is the same as the old one, there is no new data in the file.

        if not file_ready or timestamp == last_timestamp:
            continue

        # print("Mario Position: " + str(mario_position))
        # print("Enemies: " + str(enemies))
        # print("Timestamp: " + str(timestamp))

        # im = Image.open("screenshot.png")

        last_timestamp = timestamp
        file_ready = False
        sleep(0.05)
