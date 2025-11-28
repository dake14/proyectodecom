import numpy as np
import matplotlib.pyplot as plt
from generadores.canalisi import generate_canalisi

# ==============================
# 1. PARÁMETROS DEL SISTEMA
# ==============================
Nbits = 200
sps = 30
Rb = 1000
fs = Rb * sps
fc = 5000
beta = 0.25
span = 30
snr_db = 1000

phi1 = np.pi / 4
phi2 = np.pi / 2

h_chan = [
    1,
    0.3 * np.exp(1j * phi1),
    0.1 * np.exp(1j * phi2)
]
# ==============================
# 2. BITS Y MODULACIÓN BPSK
# ==============================
bits = np.random.randint(0, 2, Nbits)
symbols = 2*bits - 1

# ==============================
# 3. FILTRO RRC
# ==============================
def rrc_filter(beta, sps, span):
    N = span * sps
    t = np.arange(-N/2, N/2 + 1) / sps
    h = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - beta + (4*beta/np.pi)
        elif np.isclose(abs(ti), 1/(4*beta)):
            h[i] = (beta/np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) +
                (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
        else:
            num = (np.sin(np.pi*ti*(1-beta)) +
                   4*beta*ti*np.cos(np.pi*ti*(1+beta)))
            den = (np.pi*ti*(1-(4*beta*ti)**2))
            h[i] = num / den

    return h / np.sqrt(np.sum(h**2))

rrc = rrc_filter(beta, sps, span)

# ==============================
# 4. SOBREMUESTREO + FILTRADO TX
# ==============================
upsampled = np.zeros(len(symbols)*sps)
upsampled[::sps] = symbols
tx_bb = np.convolve(upsampled, rrc, mode='full')

# ==============================
# 5. MODULACIÓN PASABANDA
# ==============================
t = np.arange(len(tx_bb)) / fs
carrier = np.exp(1j * 2*np.pi*fc*t)
tx_pb = np.real(tx_bb * carrier)

# ==============================
# 6. CANAL + RUIDO
# ==============================

# Convolución completa del canal
tx_chan = np.convolve(tx_pb, h_chan, mode='full')

# Compensación del retardo del canal
delay_chan = len(h_chan) - 1
tx_chan = tx_chan[delay_chan:]  # Alineación correcta

# Cálculo de potencia y adición de ruido
pot_tx = np.mean(tx_chan**2)
snr_lin = 10**(snr_db/10)
pot_n = pot_tx / snr_lin
noise = np.sqrt(pot_n) * np.random.randn(len(tx_chan))

# Señal recibida pasabanda
rx_pb = tx_chan + noise


# ==============================
# 7. DEMODULACIÓN COHERENTE
# ==============================
rx_mix = rx_pb * np.exp(-1j * 2*np.pi*fc*t)
rx_bb = np.convolve(rx_mix, rrc, mode='full')

# ==============================
# 8. ALINEACIÓN Y MUESTREO
# ==============================
delay_rrc = (len(rrc) - 1) // 2
delay_total = 2 * delay_rrc

rx_bb_aligned = rx_bb[delay_total:delay_total + len(upsampled)]
tx_bb_aligned = tx_bb[delay_rrc:delay_rrc + len(upsampled)]

# Normalización independiente para comparar en misma escala
rx_bb_aligned = rx_bb_aligned / np.max(np.abs(rx_bb_aligned))
tx_bb_aligned = tx_bb_aligned / np.max(np.abs(tx_bb_aligned))

# Muestreo
muestras_simbolos = rx_bb_aligned[::sps]

# Corrección de fase
muestras_simbolos *= np.exp(-1j * np.angle(np.mean(muestras_simbolos**2)))

# Decisión
decisions = (muestras_simbolos.real >= 0).astype(int)

# ==============================
# 9. BER
# ==============================
bits_rx = decisions[:len(bits)]
num_err = np.sum(bits_rx != bits)
ber = num_err / len(bits_rx)
print(f"\nBER = {ber:.3e}   Errores = {num_err}/{len(bits_rx)}\n")

# ==============================
# 10. GRÁFICAS
# ==============================

# Comparación banda base transmitida vs recibida
plt.figure(figsize=(10,5))
plt.plot(np.real(tx_bb_aligned[:800]), label='Transmitida')
plt.plot(np.real(rx_bb_aligned[:800]), '--', label='Recibida')
plt.title("Comparación: señal transmitida vs recibida (banda base)")
plt.xlabel("Muestra")
plt.ylim(-1.1, 1.1)
plt.legend()
plt.grid()

# Señal pasabanda transmitida (eliminando transitorio)
tx_chan_cortada = tx_chan[delay_total:delay_total + len(upsampled)]

plt.figure(figsize=(10, 4))
plt.plot(tx_chan_cortada[:800])
plt.title("Pasabanda transmitida (canal + sin transitorios)")
plt.xlabel("Muestra")
plt.grid()

# Constelación
plt.figure(figsize=(5,5))
plt.scatter(muestras_simbolos.real, muestras_simbolos.imag, alpha=0.5)
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.title("Constelación BPSK")
plt.xlabel("Real")
plt.ylabel("Imag")
plt.grid()
plt.axis('equal')

# Diagrama de ojo
num_trazas = 50
samples_per_eye = 2*sps
rx_real = np.real(rx_bb_aligned)
num_muestras = (len(rx_real)//samples_per_eye)*samples_per_eye
traces = rx_real[:num_muestras].reshape((-1, samples_per_eye))

plt.figure(figsize=(6,4))
for i in range(min(num_trazas, len(traces))):
    plt.plot(traces[i], alpha=0.3)
plt.title("Diagrama de Ojo")
plt.grid()

# Espectro
from numpy.fft import fft, fftfreq, fftshift
Nfft = 4096
TXF = fftshift(fft(tx_pb[:Nfft]))
freqs = fftshift(fftfreq(Nfft, d=1/fs))

plt.figure(figsize=(7,4))
plt.plot(freqs/1000, 20*np.log10(np.abs(TXF)+1e-12))
plt.title("Espectro de la señal PASABANDA")
plt.xlabel("Frecuencia [kHz]")
plt.ylabel("Magnitud [dB]")
plt.grid()
print("Canal:", h_chan)


from scipy.signal import convolve

# Respuesta total del sistema (canal + RRC TX + RRC RX)
h_total = convolve(convolve(rrc, h_chan), rrc)
plt.figure()
plt.plot(h_total)
plt.title("Respuesta al impulso total del sistema")
plt.grid()

plt.show()
