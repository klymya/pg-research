import sqlite3
from typing import List, Tuple
import time


class TimeSeriesDatabase:
    def __init__(self, db_name: str):
        self.conn = sqlite3.connect(db_name)
        self.create_table()

    def create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS xy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            sensor_serial_number INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL
        );
        """
        create_index_sql = "CREATE INDEX IF NOT EXISTS idx_timestamp ON xy_data (timestamp);"

        with self.conn:
            self.conn.execute(create_table_sql)
            self.conn.execute(create_index_sql)

    def append_data_point(self, timestamp: int, sensor_serial_number: int, x: int, y: int):
        insert_sql = """
        INSERT INTO xy_data (timestamp, sensor_serial_number, x, y)
        VALUES (?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(insert_sql, (timestamp, sensor_serial_number, x, y))

    def query_data_points(self, start_timestamp: int, end_timestamp: int) -> List[Tuple[int, int, int, int]]:
        query_sql = """
        SELECT timestamp, sensor_serial_number, x, y
        FROM xy_data
        WHERE timestamp >= ? AND timestamp <= ?;
        """
        with self.conn:
            cursor = self.conn.execute(query_sql, (start_timestamp, end_timestamp))
            return cursor.fetchall()

    def remove_old_data_points(self, threshold_timestamp: int):
        delete_sql = """
        DELETE FROM xy_data
        WHERE timestamp < ?;
        """
        with self.conn:
            self.conn.execute(delete_sql, (threshold_timestamp,))

    def close(self):
        self.conn.close()

# Example usage
if __name__ == "__main__":
    db = TimeSeriesDatabase("xy.db")

    for n in [100, 1000, 10000]:
        timestamps = [1625097000 + i * 60 for i in range(n)]
        t_start = timestamps[-60]
        t_end = timestamps[-1]

        t0 = time.time()
        for t in timestamps:
            db.append_data_point(t, 0, 10, 10)
        t1 = time.time()

        for i in range(100):
            _ = db.query_data_points(t_start, t_end)
        t2 = time.time()

        # Remove old data points
        db.remove_old_data_points(t_end)
        t3 = time.time()

        db.remove_old_data_points(1e15)

        print(f"Number of points: {n}")
        print(f"Uppend: {(t1 - t0) / n:.03f}")
        print(f"Query: {(t2 - t1) / 100:.03f}")
        print(f"Remove: {(t3 - t2):.03f}")
        print()

    db.close()
