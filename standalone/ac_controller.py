import time
import vgamepad
import keyboard


class ACController:
    """
    A virtual controller for Assetto Corsa.
    This class uses the vgamepad library to send inputs to the game.
    """

    def __init__(self):
        """
        Initialize the virtual controller.
        """
        self.gamepad = vgamepad.VX360Gamepad()

        # Press and release a button so AC recognizes the controller
        self.gamepad.press_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()
        time.sleep(0.5)
        self.gamepad.release_button(button=vgamepad.XUSB_BUTTON.XUSB_GAMEPAD_A)
        self.gamepad.update()

    def perform(self, throttle, brake, steer):
        """
        Perform the actions in the game.
        :param throttle: The throttle value.
        :param brake: The brake value.
        :param steer: The steering value.
        """
        self.gamepad.left_trigger_float(throttle)
        self.gamepad.right_trigger_float(brake)
        self.gamepad.left_joystick_float(x_value=steer)
        self.gamepad.update()

    def reset_car(self):
        """
        Reset the car back to the starting line.
        """
        # Press the F10 key on the keyboard to trigger a respawn in the AC app
        keyboard.press('F10')
        time.sleep(0.5)
        keyboard.release('F10')
