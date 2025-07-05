import numpy as np
import os
from datetime import datetime

SAVE_DIR = "fpv_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)


def spectrum_logger(freqs, amplitude, label=0):
    """
    Сохраняет спектр (freqs + amplitude) в .npz файл с меткой класса.

    Параметры:
    - freqs: массив или список частот (в Гц)
    - amplitude: массив или список амплитуд (dB или линейно)
    - label: метка класса (0 = analog, 1 = digital, и т.д.)
    """
    if len(freqs) != len(amplitude):
        print("[!] Размерности не совпадают — спектр не сохранён.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{SAVE_DIR}/spectrum_label{label}_{timestamp}.npz"

    np.savez(
        filename,
        freq=np.array(freqs, dtype=np.float32),
        ampl=np.array(amplitude, dtype=np.float32),
        label=int(label)
    )

    print(f"[✓] Сохранено: {filename}")
