import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn, fir_filter_design, freqz
from scipy.signal.windows import hann
from scipy.fftpack import fft, fftshift
from generadores.canalisi import generate_canalisi
# Parametros del diseño
simbolos_manuales = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0]
sobremuestreo = 30
snr_db = 10
atenuacion_isi = 0.5
taps_canal = 4 
amplitud = 1
bits = simbolos_manuales
simbolos = bits * amplitud
sobre_simbolos = np.zeros(len(simbolos)*sobremuestreo)
sobre_simbolos[::sobremuestreo] = simbolos
# Canal con ISI
canal_isi = generate_canalisi(length=taps_canal, attenuation=atenuacion_isi)
tx_primitiva = np.convolve(sobre_simbolos, canal_isi, mode='same')
# RUido AWGN
ganancia = 10 **(snr_db /10)
potencia_senal = np.mean(np.abs(tx_primitiva)**2)
potencia_ruido = potencia_senal / ganancia
desviacion_std = np.sqrt(potencia_ruido)
ruido = desviacion_std * (np.random.randn(len(tx_primitiva)) + 1j * np.random.randn(len(tx_primitiva)))
tx = tx_primitiva + ruido 

plt.figure(figsize=(14, 10))

# 1. Señal sobremuestreada original
plt.subplot(4, 1, 1)
plt.plot(sobre_simbolos, label="Señal Baseband Sobremuestreada")
plt.title("Señal Sobremuestreada (Tiempo vs Muestra)")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

# 2. Canal ISI
plt.subplot(4, 1, 2)
plt.stem(canal_isi, basefmt=" ")

plt.title("Respuesta al Impulso del Canal ISI")
plt.xlabel("Taps")
plt.ylabel("Ganancia")
plt.grid(True)

# 3. Señal después del canal (ISI)
plt.subplot(4, 1, 3)
plt.plot(tx_primitiva.real, label="Señal con ISI", color='orange')
plt.title("Señal Afectada por ISI")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

# 4. Señal final con ruido AWGN
plt.subplot(4, 1, 4)
plt.plot(tx.real, label="Señal con ISI + AWGN", color='red')
plt.title("Señal Final con Ruido AWGN")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

