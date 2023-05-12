import argparse
import json
import matplotlib.animation as ani
import matplotlib.pyplot as plt
from websocket import create_connection
import numpy as np
from struct import unpack_from
from table_single_target import table_init

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


def main():

    listener = create_connection("ws://127.0.0.1:1234/")
    # Arena and sensor settings
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_PARAMS',
        'Payload': {
            'Cfg.MonitoredRoomDims': args.arena,
            'Cfg.Common.sensorOrientation.mountPlane': args.orientation,
            'Cfg.Common.sensorOrientation.transVec(3)': args.sensor_height,
            'Cfg.filterType': 'mti'
        }}))

    # Save data settings
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_PARAMS',
        'Payload': {
            'MPR.save_dir': args.save_dir,
            'MPR.read_from_file': args.read_from_file,
            'MPR.save_to_file': args.save_to_file,
            'MPR.allow_savedData_override': 0,
            'Cfg.OutputData.save_to_file': 0
        }}))

    # Set the required outputs
    listener.send(json.dumps({
        'Type': 'COMMAND',
        'ID': 'SET_OUTPUTS',
        'Payload': {
            'binary_outputs': ['targets_targetsCount', 'targets_Locs']
        }}))

    # Start
    listener.send(json.dumps({'Type': 'COMMAND', 'ID': 'START', 'Payload': {}}))
    # Request for outputs
    listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))
    # Single target struct
    target = {
        'Z': 0,
        'Buffer': [1.0, 1.0, 1.0, 1.0, 1.0]
    }

    # Pulling Data from the engine
    def update(_):
        buffer = listener.recv()
        data = to_message(buffer)

        if data['ID'] == 'BINARY_DATA':
            for val in data['Payload']:
                # If there is 1 target in the arena
                if val == 'targets_targetsCount' and data['Payload'][val] == 1:
                    # Use easier variable for target buffer
                    buf = target['Buffer']
                    # Go over all fields of outputs
                    for field in data['Payload']:

                        if field == 'targets_Locs':
                            # Calculate and update 'z' value of target
                            avg = sum(buf) / len(buf)
                            target['Z'] = ((data['Payload'][field][2]) * 0.2) + (avg * 0.8)

                            # Update target buffer with latest value and delete oldest
                            buf.append(data['Payload'][field][2])
                            buf.pop(0)

                        if field == 'targets_targetsCount':
                            # After updating target's values use it to plot the data.
                            table_cells = the_table.properties()["children"]
                            # Update the posture
                            if 0.4 >= target['Z'] > 0.01:
                                # Lying
                                table_cells[0].set_text_props(text='Target is lying')
                            else:
                                # Not lying
                                table_cells[0].set_text_props(text='Target is not lying')
                            plt.draw()
                listener.send(json.dumps({'Type': 'QUERY', 'ID': 'BINARY_DATA'}))

        # Stop script when recording finishes
        if data['ID'] == 'GET_STATUS' and (data['Payload']['status']) == 'CONFIGURING':
            exit()


    # Create the table
    the_table, fig = table_init()

    # Update figure with live data calling the update() function.
    anim = ani.FuncAnimation(fig, update, interval=0)

    plt.show()

    # Stop engine when closing table
    listener.send(json.dumps({'Type': 'COMMAND', 'ID': 'STOP', 'Payload': {}}))

    plt.close()


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--read_from_file", default=0, type=int, help="should read from recording?")
    parser.add_argument("--arena", type=float, default=[-2.5, 2.5, -2.5, 2.5, 0, 2], nargs='+',
                        help="define arena [xmin,xmax,ymin,ymax,zmin,zmax]")
    parser.add_argument("--orientation", type=str, default='xy', help="define sensor installation orientation")
    parser.add_argument("--sensor_height", type=float, default=2.6, help="define sensor installation height")
    parser.add_argument("--save_dir", type=str, default=r'C:\temp\v7.0.0', help="directory of recording/saved data")
    parser.add_argument("--save_to_file", type=int, default=0, help="define weather to save data")
    args = parser.parse_args()

    main()
