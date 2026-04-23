import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import linregress
import h5py

# Abrir arquivo .mat v7.3
with h5py.File('C:/Users/lima-/Downloads/python/MA/adt_11y_sel.mat', 'r') as f:
    lon = np.array(f['lon_sel']).squeeze()
    lat = np.array(f['lat_sel']).squeeze()
    adt = np.array(f['topog_sel'])  # deve ser (lon, lat, tempo)

# Garantir que tempo está na última dimensão
if adt.shape[-1] != 11322:
    adt = np.transpose(adt)

nutime = adt.shape[2]
datas = pd.date_range(start='1993-01-01', periods=nutime, freq='D')

# Lista dos 12 pontos (Longitude, Latitude)
pontos = [
    (-45.950745, -1.025438),
    (-45.579382,  -1.156578), 
    (-45.254512, -1.485709), 
    (-44.612493, -1.678496),
    (-44.216131, -2.334498),
    (-43.484629, -2.302975),
    (-42.735227, -2.502361),
    (-42.199938, -2.639296)
]

lonmin, lonmax = lon.min(), lon.max()
latmin, latmax = lat.min(), lat.max()
dlon = lon[1] - lon[0]
dlat = lat[1] - lat[0]

# Criar matriz para armazenar (tendências e altim1d)
slopes = []
altim_all = []

# Criar figura com 8 subplots (2 linhas x 4 colunas)
fig, axs = plt.subplots(2, 4, figsize=(24, 14))
axs = axs.flatten()

for i, (lonsel, latsel) in enumerate(pontos):
    indlonsel = int(np.floor((lonsel - lonmin) / dlon))
    indlatsel = int(np.floor((latsel - latmin) / dlat))

    altim1d = adt[indlonsel, indlatsel, :].squeeze()
    altim_all.append(altim1d)  # guarda os dados deste ponto

    dias = np.arange(nutime)

    # Ajuste linear
    slope, intercept, _, _, _ = linregress(dias, altim1d)
    slopes.append(slope)  # guarda o slope deste ponto
    altim_pol = intercept + slope * dias

    ax = axs[i]
    ax.plot(datas, altim1d, color='blue', linewidth=1.5, label='Dados')
    ax.plot(datas, altim_pol, color='red', linestyle='--', linewidth=2, label='Tendência')
    ax.grid(alpha=0.3)
    ax.set_xlim([datas[0], datas[-1]])
    ax.set_ylim([0.2, 1.2])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Ano', fontsize=12, fontweight='bold')
    ax.set_ylabel('TDA (m)', fontsize=12, fontweight='bold')

    titulo = f'MA-{i+1}: {lonsel:.2f} W, {latsel:.2f} S'
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)

    ax.text(datas[int(nutime*0.7)], altim1d.min()+0.05,
            f'Tend: {slope*365*1000:.3f} mm/ano',
            fontsize=10, color='red')

    ax.legend(fontsize=10, loc='upper left')

# Converter lista de slopes em array
slopes = np.array(slopes)

# Calcular média e desvio padrão das tendências
media_slope = np.mean(slopes)
desvio_slope = np.std(slopes)

# Mostrar no terminal
print(f"Média das tendências (slope): {media_slope:.6f} m/dia")
print(f"Desvio padrão das tendências (slope): {desvio_slope:.6f} m/dia")

# Também em mm/ano
media_slope_mm_ano = media_slope * 365 * 1000
desvio_slope_mm_ano = desvio_slope * 365 * 1000
print(f"Média das tendências (slope): {media_slope_mm_ano:.3f} mm/ano")
print(f"Desvio padrão das tendências (slope): {desvio_slope_mm_ano:.3f} mm/ano")

# Título geral da figura em negrito e maior
fig.suptitle("Séries temporais de TDA - 1993 a 2023 (8 pontos)",
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('adt_serie_temporal_8_pontos_final.png', dpi=300)
plt.show()

# Converter lista em array (8 x 11322)
altim_all = np.array(altim_all)

# Calcular média e desvio padrão geral (ao longo dos pontos)
altim_mean = np.mean(altim_all, axis=0)
altim_std = np.std(altim_all, axis=0)

# Calcular média e desvio padrão geral (valor único)
media_geral = np.mean(altim_all)
desvio_geral = np.std(altim_all)

# Mostrar no terminal
print(f"Média geral dos 8 pontos: {media_geral:.4f} m")
print(f"Desvio padrão geral dos 8 pontos: {desvio_geral:.4f} m")

# Plotar média e desvio padrão em figura separada
fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(datas, altim_mean, color='black', linewidth=2, label='Média dos 8 pontos')
ax2.fill_between(datas, altim_mean-altim_std, altim_mean+altim_std,
                 color='gray', alpha=0.3, label='±1 desvio padrão')
ax2.set_xlim([datas[0], datas[-1]])
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.tick_params(axis='x', rotation=45, labelsize=10)
ax2.set_xlabel('Ano', fontsize=12, fontweight='bold')
ax2.set_ylabel('TDA (m)', fontsize=12, fontweight='bold')
ax2.set_title("Média e desvio padrão geral - 8 pontos", fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('adt_media_desvio_8_pontos.png', dpi=300)
plt.show()


# altim_all já é (8 x 11322)
altim_all = np.array(altim_all)

# Média diária entre os 8 pontos
altim_mean = np.mean(altim_all, axis=0)

# Vetor de dias
dias = np.arange(nutime)

# Ajuste linear na série média
slope_mean, intercept_mean, r_value, p_value, slope_stderr = linregress(dias, altim_mean)

# Mostrar no terminal
print(f"Slope da série média dos 8 pontos: {slope_mean:.6f} m/dia")
print(f"Slope da série média dos 8 pontos: {slope_mean*365*1000:.3f} mm/ano")

print(f"Desvio padrão dos 8 pontos (erro do slope): {slope_stderr:.6f} m/dia")
print(f"Desvio padrão dos 8 pontos (erro do slope): {slope_stderr*365*1000:.3f} mm/ano")