import sqlite3
import time
from typing import List, Tuple


DB_NAME = "gw.db"


class XYStorage:
    def __init__(self, db_name: str | None = None) -> None:
        if db_name is None:
            db_name = DB_NAME

        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS xy_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            sensor_serial_number TEXT NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL
        );
        """
        create_index_sql_timestamp = """
        CREATE INDEX IF NOT EXISTS idx_timestamp ON xy_data (timestamp);
        """
        create_index_sql_sensor_serial_number = """
        CREATE INDEX IF NOT EXISTS idx_sensor_serial_number ON xy_data (sensor_serial_number);
        """

        with self.conn:
            self.conn.execute(create_table_sql)
            self.conn.execute(create_index_sql_timestamp)
            self.conn.execute(create_index_sql_sensor_serial_number)

    def append_data_point(self, sensor_serial_number: str, x: int, y: int, timestamp: int = None) -> None:
        if timestamp is None:
            timestamp = int(time.time())

        insert_sql = """
        INSERT INTO xy_data (timestamp, sensor_serial_number, x, y)
        VALUES (?, ?, ?, ?);
        """
        with self.conn:
            self.conn.execute(insert_sql, (timestamp, sensor_serial_number, x, y))

    def query_data_points(
            self, sensor_serial_number: str, start_timestamp: int, end_timestamp: int = None
    ) -> List[Tuple[int, int, int, int]]:
        if end_timestamp is None:
            end_timestamp = int(time.time())

        query_sql = """
        SELECT timestamp, sensor_serial_number, x, y
        FROM xy_data
        WHERE timestamp >= ? AND timestamp < ? AND sensor_serial_number = ?
        ORDER BY timestamp;
        """
        with self.conn:
            cursor = self.conn.execute(query_sql, (start_timestamp, end_timestamp, sensor_serial_number))
            return cursor.fetchall()

    def remove_old_data_points(self, threshold_min: int = 30) -> None:
        th = int(time.time()) - threshold_min * 60
        delete_sql = """
        DELETE FROM xy_data
        WHERE timestamp <= ?;
        """
        with self.conn:
            self.conn.execute(delete_sql, (th,))

    def close(self) -> None:
        self.conn.close()
