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

# Criar figura com 12 subplots (3 linhas x 4 colunas)
fig, axs = plt.subplots(2, 4, figsize=(24, 14))
axs = axs.flatten()

for i, (lonsel, latsel) in enumerate(pontos):
    indlonsel = int(np.floor((lonsel - lonmin) / dlon))
    indlatsel = int(np.floor((latsel - latmin) / dlat))

    altim1d = adt[indlonsel, indlatsel, :].squeeze()
    dias = np.arange(nutime)

    # Ajuste linear
    slope, intercept, _, _, _ = linregress(dias, altim1d)
    altim_pol = intercept + slope * dias

    ax = axs[i]
    ax.plot(datas, altim1d, color='blue', linewidth=1.5, label='Dados')
    ax.plot(datas, altim_pol, color='red', linestyle='--', linewidth=2, label='Tendência')
    ax.grid(alpha=0.3)
    ax.set_xlim([datas[0], datas[-1]])
    ax.set_ylim([0.2, 1.2])
    ax.xaxis.set_major_locator(mdates.YearLocator(5))  # marca a cada 5 anos
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=10)
    ax.set_xlabel('Ano', fontsize=12, fontweight='bold')
    ax.set_ylabel('TDA (m)', fontsize=12, fontweight='bold')

    # Título em negrito e maior
    titulo = f'MA-0{i+1}: {lonsel:.2f} W, {latsel:.2f} S'
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=12)

    # Mostrar tendência no gráfico
    ax.text(datas[int(nutime*0.7)], altim1d.min()+0.05,
            f'Tend: {slope*365*1000:.2f} mm/ano',
            fontsize=10, color='red',  fontweight='bold') # slope*365*1000:.3f

    ax.legend(fontsize=10, loc='upper left')


# Título geral da figura em negrito e maior
fig.suptitle("Séries temporais de TDA - 1993 a 2023 (12 pontos)",
             fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('adt_serie_temporal_12_pontos_melhorada.png', dpi=300)
plt.show()