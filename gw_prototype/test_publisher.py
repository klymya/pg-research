import time

import paho.mqtt.client as mqtt 

from proto.sensor_payload_pb2 import Notification, ReportingObject

broker_hostname = "localhost"
port = 1883 

def on_connect(client, userdata, flags, return_code):
    if return_code == 0:
        print("connected")
    else:
        print("could not connect, return code:", return_code)

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Client1")
# client.username_pw_set(username="user_name", password="password") # uncomment if you use password auth
client.on_connect = on_connect

client.connect(broker_hostname, port)
client.loop_start()

topic = "Test"
msg_count = 0

notification = Notification()
obj = notification.objects.add()
obj.confidence = 0.9
obj.position_x = 0.5
obj.position_y = 0.5
obj.box_x = 0.4
obj.box_y = 0.4
obj.box_w = 0.2
obj.box_h = 0.2

try:
    while msg_count < 10:
        time.sleep(1)
        msg_count += 1
        notification.timestamp = int(time.time())
        proto_msg = notification.SerializeToString()
        result = client.publish(topic, proto_msg)
        status = result[0]
        if status == 0:
            print("Message "+ str(proto_msg.hex()) + " is published to topic " + topic)
        else:
            print("Failed to send message to topic " + topic)
            if not client.is_connected():
                print("Client not connected, exiting...")
                break
finally:
    client.disconnect()
    client.loop_stop()
