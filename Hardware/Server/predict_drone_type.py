# predict_fpv.py
import torch
import numpy as np
from scipy.signal import resample
from model import DroneClassifier

# ⚠️ Замените на свои классы:
CLASSES = ["DJI", "FPV", "None"]


# === Загружаем модель один раз ===
_model = DroneClassifier(num_classes=len(CLASSES))
_model.load_state_dict(torch.load("fpv_model.pt", map_location="cpu"))
_model.eval()

def _prep_spectrum(ampl, target_len=256):
    """
    Приводим спектр к log10, нормализуем длину.
    """
    ampl = np.asarray(ampl, dtype=float)
    ampl[ampl <= 0] = 1e-10  # Защита от log(0)
    if len(ampl) != target_len:
        ampl = resample(ampl, target_len)
    spectrum = np.log10(ampl + 1e-10)
    return torch.tensor(spectrum, dtype=torch.float32)[None, None, :]  # shape: [1, 1, 256]

def predict_type(freqs, amplitude):
    """
    Предсказывает тип FPV-сигнала по power spectrum.
    freqs не используется (может быть пустым).
    """
    ps = _prep_spectrum(amplitude)
    with torch.no_grad():
        out = _model(ps)
        idx = out.argmax(1).item()
        conf = torch.softmax(out, dim=1)[0, idx].item()
    return CLASSES[idx], conf
