import os
import sys
import time
import pickle
import json
from struct import unpack_from
import argparse

import numpy as np
from websocket import create_connection


DTYPES = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.float32,
    7: np.float64,
}

ASCII_RS = '\u001e'
ASCII_US = '\u001f'

BUSY_CHARS = "/—\\|"


def to_message(buffer):
    """ parse MatNet messages from JSON / Vayyar internal binary format """
    if isinstance(buffer, str):
        return json.loads(buffer)
    seek = 0
    fields_len = unpack_from('i', buffer, seek + 4)[0]
    fields_split = unpack_from(str(fields_len) + 's', buffer, seek + 8)[0].decode('utf8').split(ASCII_RS)
    msg = {'ID': fields_split[0], 'Payload': dict.fromkeys(fields_split[1].split(ASCII_US))}
    seek += 8 + fields_len
    for key in msg['Payload']:
        seek += np.int32().nbytes
        dtype = DTYPES[np.frombuffer(buffer, np.int32, 1, seek).item()]
        seek += np.int32().nbytes
        ndims = np.frombuffer(buffer, np.int32, 1, seek).item()
        seek += np.int32().nbytes
        dims = np.frombuffer(buffer, np.int32, ndims, seek)
        seek += ndims * np.int32().nbytes
        data1 = np.frombuffer(buffer, dtype, np.prod(dims), seek)
        seek += np.prod(dims) * dtype().nbytes
        msg['Payload'][key] = data1.reshape(dims) if ndims else data1.item()
    return msg


def main(area_dims, height, opath):
    if not os.path.exists(opath):
        os.mkdir(opath)
    if len(area_dims) != 6:
        raise ValueError("Area lenght should be 6.")

    listener = create_connection("ws://127.0.0.1:1234/")
    # Arena and sensor settings
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_PARAMS',
        'Payload': {
            'Cfg.MonitoredRoomDims': area_dims,
            'Cfg.Common.sensorOrientation.mountPlane': "xz",
            'Cfg.Common.sensorOrientation.transVec(3)': height,
            'Cfg.filterType': 'mti'
        }}))

    # Save data settings
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_PARAMS',
        'Payload': {
            'MPR.save_dir': r"",
            'MPR.read_from_file': 0,
            'MPR.save_to_file': 0,
            'MPR.allow_savedData_override': 0,
            'Cfg.OutputData.save_to_file': 0
        }}))

    # Set the required outputs
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_OUTPUTS',
        'Payload': {
            'binary_outputs': [
                'targets_targetsCount',
                'targets_Locs',
                "targets_Heights",
                "targets_Velocities",
                'targets_IDs',
                'points_Locs',
                'points_absValues',
                'points_realValues',
                'points_imaginaryValues',
                'points_pointsCount',
                'targetPoints',
                'iqData_Smat_real',
                'iqData_Smat_imaginary',
                'iqData_AntPairs',
                'iqData_Freqs'
            ]
        }}))

    # Start
    listener.send(json.dumps({'Type': 'COMMAND', 'ID': 'START', 'Payload': {}}))
    # Request for outputs
    listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))

    msgs = []
    try:
        frame_count = 0
        while True:
            sys.stdout.write("\r" + "Capturing " + BUSY_CHARS[frame_count % len(BUSY_CHARS)])
            sys.stdout.flush()

            buffer = listener.recv()
            data = to_message(buffer)
            if data['ID'] == 'BINARY_DATA':
                msgs.append({"ts": time.time(), "frame": frame_count, "data": data})
                frame_count += 1
                listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
    finally:
        listener.send(json.dumps({'Type': 'COMMAND', 'ID': 'STOP', 'Payload': {}}))
        listener.close()

        print("\nData saving...")
        with open(os.path.join(opath, "data.pickle"), 'wb') as f:
            pickle.dump(msgs, f)

        with open(os.path.join(opath, "area_params.json"), 'wt') as f:
            json.dump({
                "area_dims": area_dims,
                "height": height,
                "frame_count": len(msgs)
            }, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Write data from the Vayyar radar.")
    parser.add_argument(
        "--arena", type=float, default=[-2.5, 2.5, 0.2, 5.0, 0, 2], nargs='+',
        help="Define arena [xmin, xmax, ymin, ymax, zmin, zmax]."
    )
    parser.add_argument("--height", type=float, default=1.5, help="Define sensor installation height.")
    parser.add_argument("--outpath", type=str, default='reults', help="Directory to save data.")
    args = parser.parse_args()

    main(args.arena, args.height, args.outpath)
