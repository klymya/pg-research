import tensorflow as tf


def parse_feature(string_record):
    feature = {
        'image/height': tf.io.VarLenFeature(tf.int64),
        'image/width': tf.io.VarLenFeature(tf.int64),        
        'image/source_id': tf.io.FixedLenFeature([],tf.string),
        'image/encoded': tf.io.FixedLenFeature([],tf.string),        
        'image/filename': tf.io.FixedLenFeature([],tf.string),
        'image/format': tf.io.FixedLenFeature([],tf.string),
        'image/key/sha256': tf.io.FixedLenFeature([],tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(string_record, feature)
    return example


def parse_feature_masks(string_record):
    feature = {
        'image/height': tf.io.VarLenFeature(tf.int64),
        'image/width': tf.io.VarLenFeature(tf.int64),  
        'image/source_id': tf.io.FixedLenFeature([],tf.string),
        'image/encoded': tf.io.FixedLenFeature([],tf.string),
        'image/filename': tf.io.FixedLenFeature([],tf.string),
        'image/format': tf.io.FixedLenFeature([],tf.string),
        'image/key/sha256': tf.io.FixedLenFeature([],tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        'image/encoded_categories': tf.io.FixedLenFeature([],tf.string),
        'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        'image/object/view': tf.io.VarLenFeature(tf.string),
        'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(string_record, feature)
    return example


def load_dataset(path, feature_parser):
    return tf.data.TFRecordDataset(path).map(feature_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
