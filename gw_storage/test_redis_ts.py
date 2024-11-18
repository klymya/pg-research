import json

import redis
from redis.commands.timeseries import TimeSeries

class TimeSeriesManager:
    def __init__(self, host='localhost', port=6379, retention_msecs=86400000):
        self.client = redis.Redis(host=host, port=port)
        self.retention_msecs = retention_msecs
        self.topic = 'xy'
        self._create_topic()

    def _create_topic(self):
        self.client.delete( self.topic)
        if not self.client.exists(self.topic):
            self.client.ts().create(self.topic, retention_msecs=self.retention_msecs)
            print(f"Created topic '{self.topic}' with retention policy of {self.retention_msecs} milliseconds")

    def append_data_point(self, timestamp, sensor_serial_number, x, y):
        self.client.ts().add(self.topic, timestamp, str((sensor_serial_number, x, y)))

    def query_data_points(self, start_timestamp, end_timestamp):
        data_points = self.client.ts().range(self.topic, start_timestamp, end_timestamp)
        # return json.loads(data_points)
        return data_points

    def remove_all_data(self):
        self.client.delete(self.topic)
        self._create_topic()
        print(f"Removed all data from topic '{self.topic}'")

# Example usage
if __name__ == "__main__":
    ts_manager = TimeSeriesManager()

    # Append some data points
    ts_manager.append_data_point(1638316800000, 1, 10, 20)
    ts_manager.append_data_point(1638316801000, 2, 15, 25)

    # Query data points
    ts_manager.query_data_points(1638316800000, 1638316802000)

    # Remove all data
    ts_manager.remove_all_data()
