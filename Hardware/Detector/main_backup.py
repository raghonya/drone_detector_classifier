#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import gr, blocks, fft, uhd
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
import os
from predict_fpv import predict_fpv_type
from spectrum_logger import spectrum_logger

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from datetime import datetime

SAVE_DIR = "fpv_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('spectrum_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SpectrumScanner(gr.top_block):
    def __init__(self, center_freq=1e9, samp_rate=40e6, gain=40, fft_size=2048):
        gr.top_block.__init__(self, "Spectrum Scanner")

        self.fft_size = fft_size
        self.samp_rate = samp_rate
        self.center_freq = center_freq

        self.src = uhd.usrp_source(
            "name=NI2901,num_recv_frames=128",
            uhd.stream_args(cpu_format="fc32", channels=[0])
        )
        self.src.set_samp_rate(samp_rate)
        self.src.set_center_freq(center_freq, 0)
        self.src.set_gain(gain, 0)
        self.src.set_antenna("RX2", 0)
        self.src.set_time_now(uhd.time_spec(0.1))

        self.stream_to_vector = blocks.stream_to_vector(gr.sizeof_gr_complex, fft_size)
        self.fft_block = fft.fft_vcc(fft_size, True, (), True)
        self.c2mag = blocks.complex_to_mag_squared(fft_size)
        self.vector_sink = blocks.vector_sink_c(fft_size)

        self.connect(self.src, self.stream_to_vector, self.vector_sink)

    def set_freq(self, freq):
        self.center_freq = freq
        self.src.set_center_freq(freq, 0)
        self.vector_sink.reset()
        time.sleep(0.05)

    def get_raw_vector(self):
        start_time = time.time()
        while len(self.vector_sink.data()) < self.fft_size:
            if time.time() - start_time > 1.0:
                logger.warning("Timeout waiting for samples")
                return None
            time.sleep(0.01)

        raw = np.array(self.vector_sink.data()[-self.fft_size:])
        self.vector_sink.reset()

        if np.allclose(raw, raw[0], rtol=1e-5):
            logger.warning("Flat signal detected")
            return None

        return raw

    def check_overflow(self):
        try:
            msg = self.src.get_async_msg_queue().try_pop()
            if msg and msg.metadata.error_code == uhd.uhd_types.RXMetadataErrorCode.OVERFLOW:
                logger.warning("Overflow detected")
                return True
        except:
            pass
        return False


def get_windowed_fft_power(data, fft_size):
    window = np.blackman(fft_size)
    window_power = np.sum(window ** 2)
    vec = data * window
    fft_result = np.fft.fftshift(np.fft.fft(vec))
    power = (np.abs(fft_result) ** 2) / (window_power * fft_size)
    return power

def dbfs(power):
    return 10 * np.log10(np.maximum(power, 1e-12))
def undbfs(db_vals):
    """
    Переводит массив/скаляр из dBFS обратно в линейную мощность.

    Параметры
    ---------
    db_vals : float | np.ndarray
        Значения спектра в dBFS.

    Возвращает
    ----------
    power : np.ndarray
        Линейные мощности (та же форма, что и у db_vals).
    """
    db_vals = np.asarray(db_vals, dtype=np.float64)
    db_vals = np.where(np.isfinite(db_vals), db_vals, -120.0)

    power = 10.0 ** (db_vals / 10.0)
    return np.maximum(power, 1e-12)

def save_spectrum(freqs, ampl, label=0):
    """
    Сохраняет один спектр в .npz файл
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIR}/spectrum_label{label}_{timestamp}.npz"
    np.savez(filename, freq=np.array(freqs), ampl=np.array(ampl), label=label)
    print(f"[✓] Спектр сохранён как {filename}")


def estimate_distance(pr_dbm, pt_dbm=27, gt_dbi=2, gr_dbi=5, n=2.5, freq_hz=5800e6):
    # Затухание в свободном пространстве на 1 м
    lambda_m = 3e8 / freq_hz
    pl1m = 20 * np.log10(4 * np.pi / lambda_m)

    # Модель
    distance_m = 10 ** ((pt_dbm + gt_dbi + gr_dbi - pr_dbm - pl1m) / (10 * n))
    return distance_m


def run_scan_plot(start_freq, stop_freq, step, rate, gain, fft_size, interval_s=1.0):
    tb = SpectrumScanner(center_freq=start_freq, samp_rate=rate, gain=gain, fft_size=fft_size)
    tb.start()
    logger.info("SDR stream started")

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("Частота (МГц)")
    ax.set_ylabel("Амплитуда (dBFS)")
    ax.set_title("Объединённый спектр")
    ax.grid(True)
    ax.set_ylim(bottom=-90)
    ax.set_autoscaley_on(False)
    fig.show()

    try:
        while True:
            full_freq = []
            full_ampl = []

            freq = start_freq
            while freq <= stop_freq:
                tb.set_freq(freq)

                if tb.check_overflow():
                    logger.warning("Overflow, skipping")
                    freq += step
                    continue

                raw = tb.get_raw_vector()
                if raw is None:
                    freq += step
                    continue

                spectrum_power = get_windowed_fft_power(raw, fft_size)
                spectrum_db = dbfs(spectrum_power)

                freq_axis = np.linspace(freq - rate / 2, freq + rate / 2, fft_size, endpoint=False) / 1e6

                drone, conf = predict_fpv_type(freq_axis, spectrum_power)  # <= именно spectrum_power!

                #if drone != "Other":
                if conf > 0.75 and drone != "None":

                    print(f"{drone} {conf:.3f}")
                # === Вызов предсказания

                #if(max(spectrum_db)>-70):
                #    spectrum_logger(freq_axis, spectrum_power, label=0)#DJI
                #    spectrum_logger(freq_axis, spectrum_power, label=1)#FPV
                #spectrum_logger(freq_axis, spectrum_power, label=2)#none


                full_freq.extend(freq_axis.tolist())
                full_ampl.extend(spectrum_db.tolist())


                logger.info(f"Сканирование {freq/1e6:.2f} MHz завершено")

                freq += step
                time.sleep(interval_s)

            if full_freq and full_ampl:
                try:
                    line.set_data(full_freq, full_ampl)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)
                except Exception as e:
                    logger.error(f"Ошибка обновления графика: {str(e)}")

    except KeyboardInterrupt:
        logger.info("Прервано пользователем")

    finally:
        tb.stop()
        tb.wait()
        logger.info("Ресурсы освобождены")


if __name__ == "__main__":
    CONFIG = {
        "START_FREQ": 5700e6,
        "STOP_FREQ": 5900e6,
        "STEP": 20e6,
        "RATE": 20e6,
        "GAIN": 40,
        "FFT_SIZE": 256,
        "INTERVAL_S": 0.05
    }

    run_scan_plot(
        start_freq=CONFIG["START_FREQ"],
        stop_freq=CONFIG["STOP_FREQ"],
        step=CONFIG["STEP"],
        rate=CONFIG["RATE"],
        gain=CONFIG["GAIN"],
        fft_size=CONFIG["FFT_SIZE"],
        interval_s=CONFIG["INTERVAL_S"]
    )
