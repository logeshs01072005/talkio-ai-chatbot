import subprocess
import threading

def run_streamlit():
    subprocess.run(["streamlit", "run", "app.py"])

def run_fastapi():
    subprocess.run(["uvicorn", "main:app", "--reload", "--port", "8000"])

t1 = threading.Thread(target=run_streamlit)
t2 = threading.Thread(target=run_fastapi)

t1.start()
t2.start()