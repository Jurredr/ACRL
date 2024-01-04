import pyautogui
import pytesseract
import time

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the number of times to grab the screen per second
grab_frequency = 30

# Define the duration in seconds to grab the screen
duration = 5

# Calculate the interval between each screen grab
interval = 1 / grab_frequency

# Start grabbing the screen
start_time = time.time()
end_time = start_time + duration

while time.time() < end_time:
    # Grab the screen's top left corner, and save it to a file
    screenshot = pyautogui.screenshot(region=(0, 0, 300, 400))

    # Perform OCR on the screenshot to read the text
    text = pytesseract.image_to_string(screenshot)

    # Print the text
    print(text)

    # Wait for the specified interval before grabbing the next screen
    time.sleep(interval)
