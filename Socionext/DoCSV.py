import numpy as np
import pandas as pd
import csv


def read_bin_file(file_name1, rows1, columns1):

    block_size = columns1 * 2

    with open(file_name1, "rb") as bin_file:
        # binary_data = bin_file.read(block_size)
        binary_data = bin_file.read()
        buffer_dataI = binary_data[0:block_size]
        buffer_dataQ = binary_data[block_size:2 * block_size]
        IData0 = np.frombuffer(buffer_dataI, dtype=np.uint16)
        QData0 = np.frombuffer(buffer_dataQ, dtype=np.uint16)
        # binary_data = bin_file.read(block_size)
        # IData1 = np.frombuffer(binary_data, dtype=np.uint16)
        # binary_data = bin_file.read(block_size)
        # QData1 = np.frombuffer(binary_data, dtype=np.uint16)
        # binary_data = bin_file.read(block_size)
        # IData2 = np.frombuffer(binary_data, dtype=np.uint16)
        # binary_data = bin_file.read(block_size)
        # QData2 = np.frombuffer(binary_data, dtype=np.uint16)

    IData0 = IData0.reshape((15, 512))
    QData0 = QData0.reshape((15, 512))

    return IData0, QData0


def write_csv(file_name_csv1, Data0):

    with open(file_name_csv1, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(Data0.T)


if __name__ == '__main__':
    # file_name = "c:/Database/Socionext/Socionext_27.dat"
    # file_name = "c:\\Database\\Socionext\\Maor\\empty\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_emptyI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_emptyQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\1p-static\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_1p-staticI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_1p-staticQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\1p-dynamic\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_1p-dynamicI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_1p-dynamicQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\1p-static\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_1p-staticI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_1p-staticQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\2p-static\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_2p-staticI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_2p-staticQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\2p-dynamic\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_2p-dynamicI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_2p-dynamicQ.csv"
    # file_name = "c:\\Database\\Socionext\\Maor\\3p-static\\DATASET_0.dat"
    # csv_file_nameI = "c:/Database/CSV/frame_3p-staticI.csv"
    # csv_file_nameQ = "c:/Database/CSV/frame_3p-staticQ.csv"
    file_name = "c:\\Database\\Socionext\\Maor\\3p-dynamic\\DATASET_0.dat"
    csv_file_nameI = "c:/Database/CSV/frame_3p-dynamicI.csv"
    csv_file_nameQ = "c:/Database/CSV/frame_3p-dynamicQ.csv"
    chirps_in_frame = 15
    sampled_points_in_frame = 512
    frame_size = sampled_points_in_frame * chirps_in_frame   # 7680 samples of I and Q separately each 2 bytes - 30720 bytes
    rows = 6
    columns = 7680
    IData0, QData0 = read_bin_file(file_name, rows, columns)

    # Create a DataFrame from the array
    #  df = pd.DataFrame(frame_data[2])

    write_csv(csv_file_nameI, IData0)
    write_csv(csv_file_nameQ, QData0)

