Steps:
1. Gather paths to seq files:
```bash
python collect_avi_paths.py
``` 

2. Concatenate results of `pg01.py` and `pg02.py` into `data.csv`, the file shoul have columns `seq_path` and `is_algo_test` - is file used for Algo CI test:
```bash
python combine_data.py
```

3. Check path to avi files:
```bash
python is_video_exist.py
```

4. Remove duplicates from the datalist - use `deduplication.ipynb`

5. Read annotations from seq files:
```bash
python parse_seq_files.py
```

6. Create sqlite3 database
```bash
sqlite3 people_detection.db
```

7. Add table to the db using `create_boxes_table.sql`:
```bash
sqlite3 people_detection.db < create_boxes_table.sq
```

8. Load data to db:
```bash
python import.py
```