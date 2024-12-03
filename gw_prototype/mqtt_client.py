import logging

import paho.mqtt.client as mqtt


class MQTTClient:
    def __init__(
        self, client_id: str, server: str, port: int = 1883, topics: list[tuple[str, int]] | None = None,
        on_message: str = None,
    ) -> None:
        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id)
        self._client.on_connect = self._on_connect

        self._client.connect(server, port)

        if on_message is not None and topics is not None:
            self._client.on_message = on_message
            self._client.subscribe(topics)

        self._client.loop_start()

    @staticmethod
    def _on_connect(client, userdata, flags, return_code):
        if return_code == 0:
            logging.debug(f"{client._client_id} connected")
        else:
            logging.warning("could not connect, return code:", return_code)

    def is_connected(self) -> bool:
        return self._client.is_connected()

    def push_message(self, topic: str, message: mqtt.PayloadType) -> mqtt.MQTTMessageInfo:
        return self._client.publish(topic, message)

    def close(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()
