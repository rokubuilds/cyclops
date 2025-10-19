# probe_zmq.py
import zmq, time
ctx = zmq.Context.instance()
s5 = ctx.socket(zmq.SUB); s5.connect("tcp://192.168.1.249:5555"); s5.setsockopt(zmq.SUBSCRIBE,b"")
s6 = ctx.socket(zmq.SUB); s6.connect("tcp://192.168.1.249:5556"); s6.setsockopt(zmq.SUBSCRIBE,b"")

poller = zmq.Poller()
poller.register(s5, zmq.POLLIN); poller.register(s6, zmq.POLLIN)

print("Probing… (Ctrl-C to stop)")
while True:
    events = dict(poller.poll(2000))  # 2s
    if not events:
        print("…no messages in last 2s")
        continue
    if s5 in events:
        parts = s5.recv_multipart()
        print(f"[5555] {len(parts)} part(s): sizes={[len(p) for p in parts]}")
    if s6 in events:
        parts = s6.recv_multipart()
        print(f"[5556] {len(parts)} part(s): sizes={[len(p) for p in parts]}")
