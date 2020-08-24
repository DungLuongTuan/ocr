import tensorflow as tf
import argparse
import cv2
import os


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def serialize_example(image_encoded, image_format, image_width, image_orig_width, \
                        image_class, unpadded_class, image_text):
    feature = {
        'image/encoded': _bytes_feature(image_encoded),
        'image/format': _bytes_feature(image_format),
        'image/width': _int64_feature(image_width),
        'image/orig_width': _int64_feature(image_orig_width),
        'image/class': _int64_list_feature(image_class),
        'image/unpadded_class': _int64_list_feature(unpadded_class),
        'image/text': _bytes_feature(image_text)
    }
  
    # Create a Features message using tf.train.Example.
  
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def main(args):
    null_id = 10
    max_seqlen = 13
    charset_path = args.charset_path
    save_path = args.out_path
    # read image filenames
    filenames = []
    texts = []
    with open(os.path.join(args.pad_path, "label.txt"), "r") as f:
        for row in f:
            split_row = row[:-1].split("\t")
            filenames.append(split_row[0])
            texts.append(split_row[1])
    # read character set
    charset = {}
    with open(charset_path, "r") as f:
        for row in f:
            val, key = row[:-1].split("\t")
            charset[key] = int(val)

    cnt = 0
    error_files = []
    with tf.io.TFRecordWriter(save_path) as writer:
        for i, (filename, text) in enumerate(zip(filenames, texts)):
            print(i, filename, "\t", text)
            # text = text.upper()
            if len(text) > max_seqlen:
                continue
            ### prepare all feature values
            # image/encoded
            try:
                with open(os.path.join(args.pad_path, filename), "rb") as f:
                    image_encoded = f.read()
                cnt += 1
                # image/format
                image_format = "jpg".encode()
                # image/width
                image_width = 500
                # image/orig_width
                h, w, _ = cv2.imread(os.path.join(args.unpad_path, filename)).shape
                image_orig_width = w
                # image/class
                image_class = []
                for char in text:
                    image_class.append(charset[char])
                while len(image_class) < max_seqlen:
                    image_class.append(null_id)
                # image/unpadded_class
                unpadded_class = []
                for char in text:
                    unpadded_class.append(charset[char])
                # image/text
                image_text = text.encode()
                # write to TFRecordFile
                example = serialize_example(image_encoded, image_format, image_width, image_orig_width, \
                                            image_class, unpadded_class, image_text)
                writer.write(example)
            except:
                print(filename)
            
    print()
    print(cnt)
    for file in error_files:
        print(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pad_path", help = "directory contain images", default = "data/random_ID_v1/validation")
    parser.add_argument("--unpad_path", help = "directory contain images", default = "data/random_ID_v1/validation")
    parser.add_argument("--charset_path", help = "character set file path", default = "../charsets/charset_size=11.txt")
    parser.add_argument("--out_path", help = "output path", default = "data/random_ID_v1/random_ID_v1.validation")
    args = parser.parse_args()
    main(args)

"""
    python3 gen_record.py --pad_path="data/process_extract_0606/extract_0606_resident/labeled/v2/pad_train/" --unpad_path="data/process_extract_0606/extract_0606_resident/labeled/v2/unpad_train" --out_path="data/process_extract_0606/extract_0606_resident/labeled/v2/extract_0606_resident.train"
    python3 gen_record.py --pad_path="data/process_extract_0606/extract_0606_resident/labeled/v2/pad_valid/" --unpad_path="data/process_extract_0606/extract_0606_resident/labeled/v2/unpad_valid" --out_path="data/process_extract_0606/extract_0606_resident/labeled/v2/extract_0606_resident.valid"
"""
