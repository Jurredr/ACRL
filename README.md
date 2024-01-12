# ACRL: Assetto Corsa Reinforcement Learning

## Description

A Python project that uses the [SAC algorithm](https://arxiv.org/abs/1801.01290) to learn to drive autonomously in Assetto Corsa through Reinforcement Learning.

This project is the bachelor assignment of [Jurre de Ruiter](https://www.jurre.me/) (s2580160), Technical Computer Science student at the [University of Twente](https://www.utwente.nl/en/).

## Project overview

The project consists of two main components:

1. The Assetto Corsa app (`ACRL/`), which is a Python script that runs inside Assetto Corsa and communicates real-time data with the Python project outside of Assetto Corsa.
2. The standalone Python project (`standalone/`), which is a Python script that runs outside of Assetto Corsa and uses the real-time data from the Assetto Corsa app to train a model using the SAC algorithm. It sends the model's actions back to the Assetto Corsa app through a virtual controller from [vgamepad](https://pypi.org/project/vgamepad/).

## Getting started

1. Purchase and install [Assetto Corsa](https://store.steampowered.com/app/244210/Assetto_Corsa/) on Steam, and download the free [Content Manager](https://assettocorsa.club/content-manager.html) extension software.
2. Clone this repository to your local machine. (See [here](https://help.github.com/en/articles/cloning-a-repository) for instructions on how to clone a repository.
3. Getting the Assetto Corsa app working:
    - Install the [Python 3.3.5](https://www.python.org/downloads/release/python-335/) interpreter. This is what Assetto Corsa uses and we need this locally for the socket import to not throw errors in AC. Make sure to add Python to your [PATH environment variable](https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/).
    - Copy the `ACRL` folder to the `apps/python` folder in your Assetto Corsa installation directory. (e.g. `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\apps\python`)
    - Run Assetto Corsa and enable the `ACRL` app in the `General` tab of the `Settings` menu. (You can also enable it through the `Content Manager` settings, or in the `Custom Shaders Patch` tab if you have CSP installed).
4. Getting the standalone Python project working:
    - Install a modern Python interpreter (tested on: [Python 3.10.13](https://www.python.org/downloads/release/python-31013/)). This is what we use for the part of the project that runs outside of AC.
    - Install the required Python packages by running `pip install -r requirements.txt` in the root of the `/standalone` directory.
5. Set up a new session in Assetto Corsa through Content Manager:
    - Select Practice Mode and choose track x and car y
    - Confirm tires are the default option
    - Set the number of AI opponents to 0
    - Turn on `penalties` and `ideal conditions`
    - Set controls to `Gamepad`, with speed sensitivity set to 0, steering speed set to 100%, steering gamma set to 100%, and steering filter set to 0%
6. Start the session and wait for the car to spawn on the track.
7. Run the `standalone/main.py` file to start listening for an incoming connection from Assetto Corsa.
8. Start training by clicking the `Start Training` button in the ACRL app window in Assetto Corsa. The car should start driving around the track and the model should start training. You can monitor the training progress in the console window where you started the `main.py` script.
