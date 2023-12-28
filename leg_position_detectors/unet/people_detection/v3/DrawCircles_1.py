from PIL import Image, ImageDraw
import os

radius = 17
radius_s = 7


def process_images(directory):
    # Iterate through files in the directory
    i_count = 0
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            jpg_path = os.path.join(directory, filename)
            txt_path = os.path.join(directory, os.path.splitext(filename)[0] + '.txt')

            # Check if a corresponding TXT file exists
            if os.path.isfile(txt_path):
                # Open the JPG image to get its dimensions
                with Image.open(jpg_path) as jpg_image:
                    width, height = jpg_image.size

                # Create an 8-bit PNG image with the same dimensions as the JPG
                image = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(image)

                # Read the TXT file and extract circle center coordinates
                with open(txt_path, 'r') as txt_file:
                    lines = txt_file.read().strip().split('\n')
                    for line in lines:
                        values = line.split()
                        if len(values) == 6:
                            #  foot
                            x, y = int(values[4]), int(values[5])
                            draw.ellipse((max(x - radius, 0), max(y - radius, 0), min(x + radius, width - 1), min(y + radius, height - 1)), fill=255)
                            #  upper left
                            # x, y = int(values[0]), int(values[1])
                            # draw.ellipse((max(x - radius_s, 0), max(y - radius_s, 0), min(x + radius_s, width - 1), min(y + radius_s, height - 1)), fill=255)
                            # # #  upper right
                            # x, y = int(values[0]) + int(values[2]), int(values[1])
                            # draw.ellipse((max(x - radius_s, 0), max(y - radius_s, 0), min(x + radius_s, width - 1), min(y + radius_s, height - 1)), fill=255)
                            # # #  bottom left
                            # x, y = int(values[0]), int(values[1]) + int(values[3])
                            # draw.ellipse((max(x - radius_s, 0), max(y - radius_s, 0), min(x + radius_s, width - 1), min(y + radius_s, height - 1)), fill=255)
                            # # #  bottom right
                            # x, y = int(values[0]) + int(values[2]), int(values[1]) + int(values[3])
                            # draw.ellipse((max(x - radius_s, 0), max(y - radius_s, 0), min(x + radius_s, width - 1), min(y + radius_s, height - 1)), fill=255)

                # Save the PNG image with the same name
                image.save(os.path.join(directory, os.path.splitext(filename)[0] + '.png'))
                i_count += 1
                print(i_count)
            else:
                # If no TXT file exists, create an empty PNG image with the same dimensions as the JPG
                image = Image.new('L', (width, height), 0)

                # Save the PNG image with the same name
                image.save(os.path.join(directory, os.path.splitext(filename)[0] + '.png'))
                i_count += 1
                print(i_count)


directory_path = 'c:/CamDatabase/PositionsDB/TestThin'
# directory_path = 'c:/CamDatabase/PositionsDB/TrainingThin'
# directory_path = 'c:/CamDatabase/PositionsDB/Temp'
# directory_path = 'PositionsDB/TrainingRad15'
# directory_path = 'PositionsDB/TrainingThin'
process_images(directory_path)
