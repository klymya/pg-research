CREATE TABLE boxes(
    seq_name text NOT NULL,
    seq_path text NOT NULL,
    avi_path text NOT NULL,
    frame_h integer,
    frame_w integer,
    camera_type text,
    altitude integer,
    frame_n integer,
    track_id integer,
    x_left integer,
    y_top integer,
    width integer, 
    height integer,
    pos_x integer,
    pos_y integer
);