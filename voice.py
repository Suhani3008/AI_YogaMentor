import os
import threading

# -----------------------------------
# Speak Function
# -----------------------------------

def speak(text):

    def run():

        os.system(f'say "{text}"')

    threading.Thread(
        target=run,
        daemon=True
    ).start()

