import time

import paho.mqtt.client as mqtt

from proto.sensor_payload_pb2 import Notification

def on_connect(client, userdata, flags, return_code):
    if return_code == 0:
        print("connected")
        client.subscribe("Test")
    else:
        print("could not connect, return code:", return_code)
        client.failed_connect = True


def on_message(client, userdata, message):
    notification = Notification()
    notification.ParseFromString(message.payload)
    print("Received message: ", notification.timestamp, notification.objects[0].confidence)


broker_hostname ="localhost"
port = 1883

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "Client2")
# client.username_pw_set(username="user_name", password="password") # uncomment if you use password auth
client.on_connect = on_connect
client.on_message = on_message
client.failed_connect = False

client.connect(broker_hostname, port)
client.loop_start()

# this try-finally block ensures that whenever we terminate the program earlier by hitting ctrl+c, it still gracefully exits
try:
    i = 0
    while i < 15 and client.failed_connect == False:
        time.sleep(1)
        i = i + 1
    if client.failed_connect == True:
        print('Connection failed, exiting...')

finally:
    client.disconnect()
    client.loop_stop()
