import shutil
import os
import glob


BATHCH_NUMBERS = ["_testAccuracy"]
ROOT_PTH = "/home/klymy/data/cam4_revB"
DATA_TEMPLATE = "4.2_batch{number}"
ANNOTATION_TEMPLATE = "batch{number}_corrected"
JSON_TEMPLATE = "batch{number}_corrected.json"
XML_TEMPLATE = "batch{number}_corrected.xml"
BASE_XML_NAME = "annotations.xml"
BASE_JSON_NAME = "instances_default.json"


for batch_number in BATHCH_NUMBERS:
    data_dir = os.path.join(ROOT_PTH, DATA_TEMPLATE.format(number=batch_number))
    annotation_dir = os.path.join(ROOT_PTH, ANNOTATION_TEMPLATE.format(number=batch_number))
    json_file = os.path.join(annotation_dir, JSON_TEMPLATE.format(number=batch_number))
    xml_file = os.path.join(annotation_dir, XML_TEMPLATE.format(number=batch_number))
    base_json_file = os.path.join(data_dir, BASE_JSON_NAME)
    base_xml_file = os.path.join(data_dir, BASE_XML_NAME)

    print(os.path.exists(base_json_file))
    print(os.path.exists(base_xml_file))

    os.makedirs(annotation_dir, exist_ok=True)
    shutil.copy2(base_json_file, json_file)
    shutil.copy2(base_xml_file, xml_file)
