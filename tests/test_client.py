import time, requests

URL = "http://127.0.0.1:8000/ingest"
sid = "demo2"

def post(delta, final=False):
    r = requests.post(URL, json={"session_id": sid, "text_delta": delta, "final": final})
    print(delta or "(heartbeat)", "=>", r.json())

if __name__ == "__main__":
    post("So last quarter I led a latency reduction project")
    time.sleep(0.4)
    post(" focusing on our auth service and database hotspots")
    time.sleep(1.1)  # let the pause trigger end-of-thought
    post("")         # heartbeat to fire the emit
