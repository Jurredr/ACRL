from IS_ACUtil import *

# https://pypi.org/project/vgamepad/
deltaX = 0
deltaY = 0
rad = atan2(deltaY, deltaX)  # In radians

# import vgamepad as vg
# gamepad = vg.VDS4Gamepad()
# press a button to wake the device up
# gamepad.press_button(button=vg.DS4_BUTTONS.DS4_BUTTON_TRIANGLE)
# gamepad.update()
# gamepad.left_joystick_float(x_value_float=-0.5, y_value_float=0.0)  # values between -1.0 and 1.0
# gamepad.right_joystick_float(x_value_float=-1.0, y_value_float=0.8)  # values between -1.0 and 1.0


def restart():
    ac.console("[ACRL] Respawning...")
    # Restart to session menu
    sendCMD(68)
    # Start the lap + driving
    sendCMD(69)
