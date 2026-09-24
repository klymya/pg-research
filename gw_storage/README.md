# gw_storage

A throughput spike comparing **Redis** against **SQLite** for storing the gateway's XY position
points. The deliverable is the measurement, not the code.

**Status:** ARCHIVED — frozen 2024-11. Question answered; nothing here is reused.

## Purpose

The gateway writes a stream of `(timestamp, sensor, x, y)` points and periodically queries and
expires them. The question was whether SQLite — already in the prototype — would keep up, or
whether Redis was warranted.

Three storage variants were timed on append, query and remove, at 100 / 1,000 / 10,000 points:

| Backend | Script |
|---|---|
| Redis key-value (`default`) | `test_redis.py` |
| Redis Streams | `redis_stream.py`, `test_redis_ts.py` |
| SQLite | `test_sqllite.py` |

## Result

| Points | Redis append / query / remove | SQLite append / query / remove |
|---:|---|---|
| 100 | 0.000 / 0.000 / 0.000 | 0.005 / 0.000 / 0.005 |
| 1,000 | 0.000 / 0.000 / 0.000 | 0.006 / 0.000 / 0.008 |
| 10,000 | 0.000 / 0.000 / 0.001 | 0.006 / 0.000 / 0.017 |

Seconds; raw output in `redis_res.txt` and `sqllite_res.txt`.

**Redis is faster on writes and removals — SQLite's remove cost grows with volume (5 ms → 17 ms)
while Redis stays flat. Queries are below timer resolution for both.**

The practical reading: at PG2's volumes **both are fast enough in absolute terms**, and SQLite
avoids running a Redis server on the gateway. `gw_prototype` and `PG2_GW` use SQLite. Redis
would only matter at far higher point rates than the sensor produces.

> These numbers are close to timer resolution — `0.000` means "too fast to measure here", not
> "instant". Re-measure before relying on them for a different workload.

## Connections

- Informed the storage choice in `gw_prototype/xy_storage.py` and `gw_cpp_version/XYStorage`.
- Not imported by anything; standalone.

## Dependencies

`redis` (Python client) and a Redis server via `docker-compose.yaml`. SQLite is standard
library. `xy.db` is a leftover test database.

```bash
docker compose up -d
pip install redis
```
