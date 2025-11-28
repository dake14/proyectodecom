import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, butter, freqz
from numpy.fft import fft, fftshift

# Parámetros
N = 1000                   # Número de símbolos
M = 4                      # ASK de 4 niveles
Rb = 1000                  # Tasa de bits
Fs = 10000                 # Frecuencia de muestreo
fc = 2000                  # Frecuencia de portadora
Ts = 1 / Rb                # Tiempo de símbolo
samples_per_symbol = int(Fs * Ts)
ISI_values = [0.0, 0.3, 0.6, 0.9]
SNR_dB_range = np.arange(0, 21, 5)
amplitudes = np.linspace(1, M, M)

# Señal baseband ASK multinivel
def generate_ask_symbols(N, M):
    symbols = np.random.randint(0, M, N)
    levels = np.linspace(-1, 1, M)  # Niveles balanceados
    return levels[symbols], levels

# Aplicar ISI con un filtro simple
def apply_isi(signal, isi_factor):
    h = np.array([1, isi_factor])
    return lfilter(h, [1], signal)

# Modulador pasabanda ASK
def modulate_passband(symbols, fc, Fs, samples_per_symbol):
    t = np.arange(len(symbols) * samples_per_symbol) / Fs
    upsampled = np.zeros(len(t))
    upsampled[::samples_per_symbol] = symbols
    carrier = np.cos(2 * np.pi * fc * t)
    return upsampled * carrier, t

# Añadir ruido blanco
def add_awgn(signal, SNR_dB):
    signal_power = np.mean(signal ** 2)
    SNR_linear = 10 ** (SNR_dB / 10)
    noise_power = signal_power / SNR_linear
    noise = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise

# Gráfica de constelación
def plot_constellation(symbols, title="Constelación"):
    plt.figure()
    plt.scatter(symbols, np.zeros_like(symbols), c='red')
    plt.title(title)
    plt.xlabel("Amplitud")
    plt.grid(True)
    plt.show()

# Espectro de la señal
def plot_spectrum(signal, Fs, title="Espectro"):
    plt.figure()
    freqs = np.linspace(-Fs/2, Fs/2, len(signal))
    spectrum = np.abs(fftshift(fft(signal)))
    plt.plot(freqs, spectrum)
    plt.title(title)
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.grid(True)
    plt.show()

# Señal en el tiempo
def plot_signal_time(signal, t, title="Señal en el tiempo"):
    plt.figure()
    plt.plot(t[:1000], signal[:1000])
    plt.title(title)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    plt.grid(True)
    plt.show()

# Calcular SNR real
def calculate_snr(signal, noisy_signal):
    noise = noisy_signal - signal
    return 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))

# Matriz de SNR vs. ISI
results = np.zeros((len(ISI_values), len(SNR_dB_range)))

for i, isi in enumerate(ISI_values):
    for j, snr in enumerate(SNR_dB_range):
        baseband_symbols, levels = generate_ask_symbols(N, M)
        symbols_isi = apply_isi(baseband_symbols, isi)
        modulated, t = modulate_passband(symbols_isi, fc, Fs, samples_per_symbol)
        received = add_awgn(modulated, snr)
        actual_snr = calculate_snr(modulated, received)
        results[i, j] = actual_snr

# Mostrar matriz de resultados
import pandas as pd
import seaborn as sns

df = pd.DataFrame(results, index=[f"ISI={x:.1f}" for x in ISI_values],
                  columns=[f"SNR_in={x}dB" for x in SNR_dB_range])

plt.figure(figsize=(10, 6))
sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis")
plt.title("SNR Real vs. ISI y SNR_in")
plt.show()

# Graficar constelación
baseband_symbols, _ = generate_ask_symbols(200, M)
plot_constellation(baseband_symbols, "Constelación ASK M=4")

# Graficar señal pasabanda
symbols_isi = apply_isi(baseband_symbols, isi_factor=0.5)
modulated_signal, t = modulate_passband(symbols_isi, fc, Fs, samples_per_symbol)
plot_signal_time(modulated_signal, t, "Señal Pasabanda con ISI")

# Graficar espectro
plot_spectrum(modulated_signal, Fs, "Espectro de la Señal Pasabanda")
