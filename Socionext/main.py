
def config_parameters_socionext():
    c = 3e8  # speed of light m/s

    # For d_max == 6 m, d_res is 0.022 m, for 12 m - 0.044 m
    # For 12 m bandwidth BW must be two times smaller, otherwise will not be achieved Nyquist rate
    d_max = 6  # m
    #  d_max = 12  # m
    Tc = 1100e-6  # time of chirp, sec
    BW = 3.4e9  # bandwidth Hz
    Central_freq = 60.5e9
    if d_max == 12:
        BW = 3.4e9

    print("d_max ", d_max, " m")
    print("Tc ", Tc * 1000000, " microsec")
    N_chirps_frame = 31
    f_lo = Central_freq - BW / 2  # start frequency
    print("start frequency ", f_lo)
    f_xtl = 40e6  # max ADC frequency

    S = BW / Tc
    print("S ", S / 1e9, " GHz/s")
    lyambda = c / f_lo
    v_max = lyambda / (4 * Tc)
    print("v_max ", v_max, " m/s")
    d_res = c / (2 * BW)
    print("d_res ", d_res, " m")
    Tf = N_chirps_frame * Tc
    v_res = lyambda / (2 * Tf)
    print("v_res ", v_res, " m/s")
    print("Tf ", Tf, " sec")
    F_if_max = 2 * S * d_max / c
    print("F_if_max ", F_if_max / 1000, " KHz")
    div_table5_3 = 160   #  512 points per chip for Tc = 4400e-6 (d_max = 6 m) and 1024 for Tc = 440e-6 (d_max = 12 m)
    fs_IQ = f_xtl / div_table5_3  # for 512 points per chirp, table. 5-3
    print("fs_IQ ", fs_IQ / 1000, " KHz")
    print("n ", 256)
    print("N_chirps_frame ", N_chirps_frame)


def config_parameters_socionext1():
    c = 3e8  # speed of light m/s
    f_xtl = 40e6  # max ADC frequency
    Rframe = 1  # fps
    coef_Tc_Tceff = 1.074
    Tw = 30e-6  #   Wait time, table 5-5

    # Mode 7 d_max = 5.15, d_res = 0.04, v_max = 7.6, v_res = 0.5
    Tc = 110e-6
    BW = 4e9  # bandwidth Hz
    n = 256
    N_chirps_frame = 31
    div_table5_3 = 16   #  256 points per chip for Tc = 110-6 and 1024 for Tc = 440e-6 (d_max = 12 m)

    # Mode 8 d_max = 8.25, d_res = 0.06, v_max = 7.53, v_res = 0.5
    # Tc = 110e-6
    # BW = 2.5e9  # bandwidth Hz
    # n = 256
    # N_chirps_frame = 31
    # div_table5_3 = 16   #  256 points per chip for Tc = 110-6 and 1024 for Tc = 440e-6 (d_max = 12 m)

    # d_max = 7, d_res = 0.027, v_max = 4.34, v_res = 0.58 Tf = 0.0033 s
    # Tc = 220e-6
    # BW = 5.9e9  # bandwidth Hz
    # n = 512
    # N_chirps_frame = 15
    # div_table5_3 = 16   #  512 points per chip for Tc = 220-6

    # d_max = 7, d_res = 0.027, v_max = 2.31, v_res = 0.31, Tf = 0.0066 s
    # Tc = 440e-6
    # BW = 5.9e9  # bandwidth Hz
    # n = 512
    # N_chirps_frame = 15
    # div_table5_3 = 32   #  512 points per chip for Tc = 220-6

    # d_max = 7, d_res = 0.0557, v_max = 2.25, v_res = 0.145 Tf = 0.014 s  Second best ?
    # Tc = 440e-6
    # BW = 2.95e9  # bandwidth Hz
    # n = 256
    # N_chirps_frame = 31
    # div_table5_3 = 64

    # d_max = 7, d_res = 0.027, v_max = 0.96, v_res = 0.13, Tf = 0.0165 s  Best ?
    # Tc = 1100e-6
    # BW = 5.9e9  # bandwidth Hz
    # n = 512
    # N_chirps_frame = 15
    # div_table5_3 = 80   #  512 points per chip for Tc = 220-6

    d_res = c * coef_Tc_Tceff / (2 * BW)
    d_max = d_res * n / 2
    print("d_res ", d_res, "d_max ", d_max)


    # d_max = 6
    tau = 2 * d_max / c
    print("tau ", tau * 1000000, "microseconds")
    Cc = 1 / ((Tc + Tw) * 1.2)  #   5.3.2.7, 1 to 8192, table 5-2
    # if Cc > N_chirps_frame:
    #     Cc = N_chirps_frame
    print("Cc ", Cc)
    # BW = 6.8e9  # bandwidth Hz
    # BW = 5.9e9  # bandwidth Hz
    Central_freq = 60.5e9
    f_lo = Central_freq - BW / 2  # start frequency
    print("start frequency ", f_lo/1e9)
    lyambda = c / f_lo
    v_max = lyambda / (4 * Tc)
    print("v_max ", v_max, " m/s")
    v_max = lyambda * Rframe * Cc / 4
    print("v_max ", v_max, " m/s")
    S = BW / Tc
    Tceff = Tc / coef_Tc_Tceff  #  table 5.14
    print("Tc Tceff ", Tc, Tceff)
    F_if_max = S * tau
    print("F_if_max ", F_if_max / 1000, " KHz")
    # n = F_if_max * Tceff * 2
    # print("n first", n)
    # n = 4 * d_max * BW / (c * coef_Tc_Tceff)
    # print("n calculated", n)
    # Tceff = n * 16 / f_xtl
#    F_if_max = n / (Tc * 2)
#    print("F_if_max ", F_if_max / 1000, " KHz")
    F_if_max = 2 * S * d_max / c
    print("F_if_max ", F_if_max / 1000, " KHz")

    fs_IQ = f_xtl / div_table5_3  # for 512 points per chirp, table. 5-3
    print("fs_IQ ", fs_IQ / 1000, " KHz")
    Tf = N_chirps_frame * Tc
    v_res = lyambda / (2 * Tf)
    print("v_res ", v_res, " m/s")
    v_res = lyambda * Rframe * Cc / (2 * N_chirps_frame)
    print("v_res ", v_res, " m/s")
    print("Tf ", Tf, " sec")


if __name__ == '__main__':

    config_parameters_socionext1()


