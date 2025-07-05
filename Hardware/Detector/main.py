#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gnuradio import gr, blocks, fft, uhd
import numpy as np
import asyncio
import websockets
import json
import logging
import time
from predict_fpv import predict_fpv_type
import matplotlib.pyplot as plt



# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        #logging.FileHandler('spectrum_log.txt'),
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
        time.sleep(0.01)

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


async def websocket_sender(websocket, message):
    try:
        await websocket.send(message)
    except Exception as e:
        logger.error(f"Send failed: {str(e)}")
        raise


async def continuous_scan_and_send(
    uri, start_freq, stop_freq, step, rate, gain, fft_size, interval_s=1.0
):
    tb = None
    websocket = None


    # Подготовка графика
    # plt.ion()
    # fig, ax = plt.subplots()
    # line, = ax.plot([], [])
    # ax.set_xlabel("Частота (МГц)")
    # ax.set_ylabel("Амплитуда (dBFS)")
    # ax.set_title("Объединённый спектр")
    # ax.grid(True)
    # ax.set_ylim(bottom=-90)  # фиксируем минимум по Y
    # ax.set_autoscaley_on(False)  # отключаем автоскейл только по Y
    #
    # fig.show()

    full_freq = []
    full_ampl = []

    try:
        websocket = await websockets.connect(uri)
        logger.info(f"Connected to WebSocket server: {uri}")

        # Send initial frequency limits
        limits_payload = json.dumps({
            "CMD": "Limits",
            "Payload": json.dumps({
                "Min": round(start_freq /1e6, 2),
                "Max": round(stop_freq /1e6, 2)
            })
        })

        await websocket_sender(websocket, limits_payload)
        logger.info("Sent frequency limits")

        tb = SpectrumScanner(center_freq=start_freq, samp_rate=rate, gain=gain, fft_size=fft_size)
        tb.start()
        logger.info("SDR stream started")

        while True:
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

                # Сбор для графика
                full_freq.extend(freq_axis.tolist())
                full_ampl.extend(spectrum_db.tolist())

                # Округлить до 2 знаков
                amplitude_rounded = [round(val, 2) for val in spectrum_db]

                drone_type, conf = predict_fpv_type(freq_axis, spectrum_power)  # <= именно spectrum_power!
                if(drone_type!="None"):
                    if drone_type == "FPV":
                        if(conf<0.5) :drone_type =""
                    elif drone_type == "DJI":
                        if(conf<0.7) :drone_type =""



                print(f"{drone_type} {conf:.3f}")
                # WebSocket отправка
                payload = json.dumps({
                    "Frequency": freq_axis.tolist(),
                    "Amplitude": amplitude_rounded,
                    "Type": drone_type
                })
                message = json.dumps({
                    "CMD": "Data",
                    "Payload": payload
                })

                await websocket_sender(websocket, message)
                logger.info(f"Sent spectrum for {freq/1e6:.2f} MHz")

                freq += step
                await asyncio.sleep(interval_s)

            # Обновление графика после прохода по диапазону
            if full_freq and full_ampl:
                try:
                    pass
                    # line.set_data(full_freq, full_ampl)
                    # ax.relim()
                    # ax.autoscale_view()
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()
                    #plt.pause(0.001)
                except Exception as e:
                    logger.error(f"Graph update error: {str(e)}")
                full_freq.clear()
                full_ampl.clear()

    except KeyboardInterrupt:
        logger.info("Terminated by user")

    except Exception as e:
        logger.error(f"Error: {str(e)}. Restarting in 5 seconds...")
        await asyncio.sleep(5)

    finally:
        if tb:
            tb.stop()
            tb.wait()
        if websocket:
            await websocket.close()
            logger.info("WebSocket connection closed")
        logger.info("Resources released")


if __name__ == "__main__":
    CONFIG = {
        "URI": "ws://192.168.16.151:7000/data",
        "START_FREQ": 5700e6,
        "STOP_FREQ": 5900e6,
        "STEP": 20e6,
        "RATE": 20e6,
        "GAIN": 40,
        "FFT_SIZE": 256,
        "INTERVAL_S": 0.05
    }

    while True:
        try:
            asyncio.run(
                continuous_scan_and_send(
                    uri=CONFIG["URI"],
                    start_freq=CONFIG["START_FREQ"],
                    stop_freq=CONFIG["STOP_FREQ"],
                    step=CONFIG["STEP"],
                    rate=CONFIG["RATE"],
                    gain=CONFIG["GAIN"],
                    fft_size=CONFIG["FFT_SIZE"],
                    interval_s=CONFIG["INTERVAL_S"]
                )
            )
        except KeyboardInterrupt:
            logger.info("Application terminated by user.")
            break
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}. Retrying in 5 seconds...")
            time.sleep(5)

