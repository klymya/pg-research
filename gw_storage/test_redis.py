import redis
import json
from datetime import datetime, timedelta


class TimeSeriesStore:
    def __init__(self, host='localhost', port=6379, retention_period=86400):
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.topic = 'xy'
        self.retention_period = retention_period  # in seconds (1 day = 86400 seconds)
        self._create_topic_with_retention()

    def _create_topic_with_retention(self):
        # Set the retention policy for the topic
        self.client.config_set('maxmemory-policy', 'allkeys-lru')
        self.client.config_set('maxmemory', '100mb')  # Set max memory limit for Redis
        self.client.config_set('maxmemory-samples', 5)  # Set the number of samples to check for eviction

    def append_data_point(self, timestamp, sensor_serial_number, x, y):
        data_point = {
            'timestamp': timestamp,
            'sensor_serial_number': sensor_serial_number,
            'x': x,
            'y': y
        }
        self.client.zadd(self.topic, {json.dumps(data_point): timestamp})

    def query_data_points(self, start_timestamp, end_timestamp):
        data_points = self.client.zrangebyscore(self.topic, start_timestamp, end_timestamp)
        return [json.loads(data_point) for data_point in data_points]

    def remove_old_data_points(self):
        cutoff_timestamp = int((datetime.now() - timedelta(seconds=self.retention_period)).timestamp())
        self.client.zremrangebyscore(self.topic, '-inf', cutoff_timestamp)

    def clear_all_data(self):
        self.client.delete(self.topic)

# Example usage
if __name__ == "__main__":
    store = TimeSeriesStore()

    # Append new data points
    store.append_data_point(1633036800, 123, 10, 20)
    store.append_data_point(1633123200, 124, 15, 25)

    # Query data points
    data_points = store.query_data_points(1633036800, 1633123200)
    print("Queried Data Points:", data_points)

    # Remove old data points
    store.remove_old_data_points()

    # Clear all data
    store.clear_all_data()


# if __name__ == "__main__":
#     import time

#     store = TimeSeriesStore()

#     for n in [100, 1000, 10000]:
#         timestamps = [1625097000 + i * 60 for i in range(n)]
#         t_start = timestamps[-60]
#         t_end = timestamps[-1]

#         t0 = time.time()
#         for t in timestamps:
#             store.append_data_point(t, 0, 10, 10)
#         t1 = time.time()

#         for i in range(100):
#             _ = store.query_data_points(t_start, t_end)
#         t2 = time.time()

#         # Remove old data points
#         store.remove_old_data_points()
#         t3 = time.time()

#         # Clear all data
#         store.clear_all_data()

#         print(f"Number of points: {n}")
#         print(f"Uppend: {(t1 - t0) / n:.03f}")
#         print(f"Query: {(t2 - t1) / 100:.03f}")
#         print(f"Remove: {(t3 - t2):.03f}")
#         print()
