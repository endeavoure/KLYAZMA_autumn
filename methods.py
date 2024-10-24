import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate
from scipy.fft import ifft, fft, fftfreq, fftshift, ifft2, fft2


def load_file(path: str, name: str, dct: dict):
    sample_rate, data = wavfile.read(path)
    dct['name'].append(name)
    dct['sample_rate'].append(sample_rate)
    dct['data'].append(data)
    time = np.arange(len(data)) / sample_rate
    dct['time'].append(time)


def load_dataframe(paths: list, names: list, dct: dict):
    for i in range(len(paths)):
        load_file(path=paths[i], name=names[i], dct=dct)
    df = pd.DataFrame(data=dct)

    return df


def plot_signal(data: np.ndarray, time: np.ndarray, name: str):
    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.xlabel('Время, с')
    plt.ylabel('Амплитуда, у.е.')
    plt.title(name)
    plt.grid(True)
    plt.show()


def cut_signal(index: int, borders: list, df: pd.DataFrame):
    beg = borders[0]
    end = borders[1]
    beg_index = df['sample_rate'][index] * beg
    end_index = df['sample_rate'][index] * end

    return df['data'][index][beg_index:end_index], df['time'][index][beg_index:end_index], df['name'][index]


def cut_signal_by_time(index: int, borders: list, df: pd.DataFrame):
    sr = df['sample_rate'][0]
    ind_borders = [x * sr for x in borders]
    ind_borders = [np.floor(ind_borders[0]), np.ceil(ind_borders[1])]
    ind_borders = [int(x) for x in ind_borders]

    return df['data'][index][ind_borders[0]:ind_borders[1]], df['time'][index][ind_borders[0]:ind_borders[1]], df['name'][index]


def plot_cut(borders: list, number: int, df: pd.DataFrame):
    dct = {'data': [], 'time': [], 'name': []}
    for i in range(number):
        data, time, name = cut_signal_by_time(index=i, borders=borders, df=df)
        dct['data'].append(data)
        dct['time'].append(time)
        dct['name'].append(name)
    df_cut = pd.DataFrame(data=dct)

    for i in range(len(df_cut['data'])):
        plot_signal(data=df_cut['data'][i]/max(df_cut['data'][i]),
                    time=df_cut['time'][i], name=df_cut['name'][i])
        print(
            f"Максимум по амплитуде: {max(df_cut['data'][i])/max(df_cut['data'][i])}")
        print(
            f"Время максимума: {df_cut['time'][i][np.argmax(df_cut['data'][i])]} с.")

    return df_cut


def corr_f(signal1: np.ndarray, signal2: np.ndarray, sample_rate: int, filter_freq: int, do_filter: bool):
    min_len = min(len(signal1), len(signal2))
    data1 = ifft(signal1)
    data2 = ifft(signal2)

    if do_filter == True:
        ser1 = np.zeros(min_len)
        ser2 = np.zeros(min_len)
        freq = fftfreq(min_len, 1 / sample_rate)

        for i in range(0, min_len):
            if freq[i] > -filter_freq:
                if freq[i] < filter_freq:
                    ser1[i] = data1[i]
                    ser2[i] = data2[i]
        corr = fft((ser1)*np.conj(ser2))
    else:
        corr = fft((data1)*np.conj((data2)))
        corr = fftshift(corr)

    if len(corr) % 2 == 0:
        lags = np.arange(-len(corr)/2, len(corr)/2)
        lags = lags/sample_rate
    else:
        lags = np.arange(-(len(corr)-1)/2, len(corr)/2)
        lags = lags/sample_rate

    return corr, lags


def split_array(arr: np.ndarray, n: int):
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def mean_corr(df: pd.DataFrame, pair: list, n: int):
    chunks_time = split_array(df['time'][0], n)
    chunks_signal_one = split_array(df['data'][0], n)
    chunks_signal_two = split_array(df['data'][1], n)
    chunks_signal_three = split_array(df['data'][2], n)

    corr_lst = []
    for i in range(len(chunks_signal_one)):
        if pair == [1, 2] or pair == [2, 1]:
            corr, lags = corr_f(
                chunks_signal_one[i], chunks_signal_two[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)
        elif pair == [1, 3] or pair == [3, 1]:
            corr, lags = corr_f(
                chunks_signal_one[i], chunks_signal_three[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)
        elif pair == [2, 3] or pair == [3, 2]:
            corr, lags = corr_f(
                chunks_signal_two[i], chunks_signal_three[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)
        elif pair == [1, 1]:
            corr, lags = corr_f(
                chunks_signal_one[i], chunks_signal_one[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)
        elif pair == [2, 2]:
            corr, lags = corr_f(
                chunks_signal_two[i], chunks_signal_two[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)
        elif pair == [3, 3]:
            corr, lags = corr_f(
                chunks_signal_three[i], chunks_signal_three[i], sample_rate=44100, filter_freq=200, do_filter=False)
            corr_lst.append(corr)

    corr_lst_abs = []
    for i in range(len(corr_lst)):
        for j in range(len(corr_lst[i])):
            corr_lst_abs.append(np.abs(corr_lst[i][j]))

    corr_lst_abs = split_array(corr_lst_abs, n)

    return [np.mean(x) for x in corr_lst_abs]
