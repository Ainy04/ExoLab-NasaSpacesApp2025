import subprocess
import threading
import time
import webbrowser
from pathlib import Path
import os

FLASK_FILE = "app.py"
STREAMLI_FILE = "dashboard.py"
FLASK_PORT = 5000
STREAMLI_PORT = 8501

def run_flask():
    subprocess.run(['python', FLASK_FILE])

def run_streamlit():
    subprocess.run(["streamlit", "run", STREAMLI_FILE, "--server.port", str(STREAMLI_PORT)])

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    time.sleep(4)


    stream_thread = threading.Thread(target=run_streamlit)
    stream_thread.start()

    time.sleep(4)

webbrowser.open(f"http://localhost:{STREAMLI_PORT}")

flask_thread.join()
stream_thread.join()    