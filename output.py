from IS_ACUtil import *  # noqa: E402


def restart():
    ac.console("[ACRL] Respawning...")
    # Restart to session menu
    sendCMD(68)
    # Start the lap + driving
    sendCMD(69)
