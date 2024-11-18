import redis
import time

class TimeSeriesStore:
    def __init__(self, host='localhost', port=6379, retention_ms=86400000):
        self.client = redis.StrictRedis(host=host, port=port, decode_responses=True)
        self.stream_name = 'xy'
        self.retention_ms = retention_ms

    def append_data_point(self, timestamp, sensor_serial_number, x, y):
        self.client.xadd(self.stream_name, {
            'timestamp': timestamp,
            'sensor_serial_number': sensor_serial_number,
            'x': x,
            'y': y
        }, id=f"{timestamp}-0", maxlen=1000)

    def query_data_points(self, start_timestamp, end_timestamp):
        start_id = f'{start_timestamp}-0'
        end_id = f'{end_timestamp}-0'
        return self.client.xrange(self.stream_name, min=start_id, max=end_id)

    def remove_all_data(self):
        self.client.delete(self.stream_name)

# Example usage
# if __name__ == "__main__":
#     store = TimeSeriesStore()

#     # Append some data points
#     current_time = int(time.time())
#     store.append_data_point(current_time, 123, 10, 20)
#     store.append_data_point(current_time + 1, 124, 15, 25)

#     # Query data points
#     start_time = current_time - 1000
#     end_time = current_time + 1000
#     data_points = store.query_data_points(start_time, end_time)
#     for data_point in data_points:
#         print(data_point)

#     # Remove all data
#     store.remove_all_data()
if __name__ == "__main__":
    store = TimeSeriesStore()

    for n in [100, 1000, 10000]:
        timestamps = [1625097000 + i * 60 for i in range(n)]
        t_start = timestamps[-60]
        t_end = timestamps[-1]

        t0 = time.time()
        for t in timestamps:
            store.append_data_point(t, 0, 10, 10)
        t1 = time.time()

        for i in range(100):
            _ = store.query_data_points(t_start, t_end)
        t2 = time.time()

        # Remove old data points
        store.remove_all_data()
        t3 = time.time()

        print(f"Number of points: {n}")
        print(f"Uppend: {(t1 - t0) / n:.03f}")
        print(f"Query: {(t2 - t1) / 100:.03f}")
        print(f"Remove: {(t3 - t2):.03f}")
        print()
