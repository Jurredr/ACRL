import time
import vgamepad
import keyboard
import numpy as np


class ACController:
    """
    A virtual controller for Assetto Corsa.
    This class uses the vgamepad library to send inputs to the game.
    """

    def __init__(self, steer_scale=[-360, 360]):
        """
        Initialize the virtual controller.
        """
        self.steer_scale = steer_scale
        self.gamepad = vgamepad.VX360Gamepad()

        # Press and release a button so AC recognizes the controller
        self.gamepad.press_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)
        self.gamepad.release_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()

    def perform(self, throttle_brake, steer):
        """
        Perform the actions in the game.
        :param throttle: The throttle value.
        :param brake: The brake value.
        :param steer: The steering value.
        """
        throttle = max(0.0, throttle_brake)
        brake = max(0.0, -throttle_brake)
        self.gamepad.left_trigger_float(value_float=brake)
        self.gamepad.right_trigger_float(value_float=throttle)
        self.gamepad.left_joystick_float(
            x_value_float=steer, y_value_float=0.0)
        self.gamepad.update()
        steer_degrees = np.interp(steer, [-1.0, 1.0], self.steer_scale)

    def reset_car(self):
        """
        Reset the car back to the starting line.
        """
        # Press the F10 key on the keyboard to trigger a respawn in the AC app
        keyboard.press('F10')
        time.sleep(0.5)
        keyboard.release('F10')

        # Move the car forward a little bit so it's over the starting line
        self.perform(0.0, 0.0)
        time.sleep(0.4)
        self.perform(1.0, 0.0)
        time.sleep(2.4)
        self.perform(-1.0, 0.0)
        time.sleep(1.4)
        self.perform(0.0, 0.0)
        time.sleep(0.3)