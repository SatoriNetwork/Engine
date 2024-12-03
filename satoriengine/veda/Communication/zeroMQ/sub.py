import zmq
import sys

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")

topic_filter = sys.argv[1] if len(sys.argv) > 1 else "Satori"
socket.setsockopt_string(zmq.SUBSCRIBE, topic_filter)

while True:
    message = socket.recv_string()
    topic, messagedata = message.split(" ", 1)
    print(f"Received message on topic {topic}: {messagedata}")
