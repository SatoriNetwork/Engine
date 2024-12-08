import zmq
import time
import random

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5555")

while True:
    topic = topic = random.choice(["Krishna", "Satori", "Jordan"])
    message = f"Message for topic {topic}"
    socket.send_string(f"{topic} {message}")
    print(f"Sent: {topic} {message}")
    time.sleep(1)
