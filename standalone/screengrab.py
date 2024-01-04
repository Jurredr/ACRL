import pyautogui
import pytesseract
import time

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Start grabbing the screen
start_time = time.time()

# Grab the screen's top left corner, and save it to a file
screenshot = pyautogui.screenshot(region=(0, 0, 300, 400))

# Perform OCR on the screenshot to read the text
text = pytesseract.image_to_string(screenshot)

# Print the text
end_time = time.time()
print(end_time * 1000 - start_time * 1000, "ms")
