import sys
import os

import time
import cv2

BUSY_CHARS = "/—\\|"


def main(outname):
    if not os.path.isdir(outname):
        os.mkdir(outname)

    video_path = os.path.join(outname, "capture.avi")
    ts_path = os.path.join(outname, "timestamps.txt")

    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    print(int(cap.get(cv2.CAP_PROP_FPS)), frame_width, frame_width)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        int(cap.get(cv2.CAP_PROP_FPS)),
        (frame_width, frame_height)
    )

    ts_file = open(ts_path, 'wt')
    try:
        frame_count = 0
        while(True):
            sys.stdout.write("\r" + "Capturing " + BUSY_CHARS[frame_count % len(BUSY_CHARS)])
            sys.stdout.flush()
            frame_count += 1

            ret, frame = cap.read()
            ts = time.time()

            if ret is True:

                # Write the frame into the file 'output.avi'
                out.write(frame)
                ts_file.write("{}\n".format(ts))
            # Break the loop
            else:
                break

    finally:
        # When everything done, release the video capture and video write objects
        cap.release()
        out.release()
        ts_file.close()


if __name__ == "__main__":
    main(sys.argv[1])
