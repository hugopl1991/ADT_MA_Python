import numpy as np
import h5py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# ==========================================================
# Leitura do arquivo .mat (v7.3)
# ==========================================================
with h5py.File(
    r'C:/Users/lima-/Downloads/python/MA/adt_11y_sel.mat', 'r'
) as f:
    adt_raw = np.array(f['topog_sel'])
    lon = np.array(f['lon_sel']).squeeze()
    lat = np.array(f['lat_sel']).squeeze()

# ==========================================================
# Garantir ordem (lon, lat, tempo)
# ==========================================================
if adt_raw.shape[0] == lon.size and adt_raw.shape[1] == lat.size:
    adt = adt_raw
elif adt_raw.shape[-1] == lon.size and adt_raw.shape[-2] == lat.size:
    adt = np.transpose(adt_raw, (2, 1, 0))
else:
    adt = np.transpose(adt_raw, (1, 0, 2))

nulon, nulat, nutime = adt.shape

# Grade
LON, LAT = np.meshgrid(lon, lat, indexing='xy')

# Tempo (dias)
t = np.arange(nutime, dtype=float)

# ==========================================================
# Pré-alocação
# ==========================================================
adt_media = np.full((nulon, nulat), np.nan)
adt_despa = np.full((nulon, nulat), np.nan)
adt_coef = np.full((nulon, nulat), np.nan)
adt_rmse = np.full((nulon, nulat), np.nan)

# ==========================================================
# Estatísticas ponto a ponto
# ==========================================================
for i in range(nulon):
    for j in range(nulat):
        serie = adt[i, j, :]
        mask = ~np.isnan(serie)

        if mask.sum() < 2:
            continue

        adt_media[i, j] = np.nanmean(serie)
        adt_despa[i, j] = np.nanstd(serie)

        p = np.polyfit(t[mask], serie[mask], 1)
        y_est = np.polyval(p, t[mask])
        res = serie[mask] - y_est

        adt_coef[i, j] = p[0]
        adt_rmse[i, j] = np.sqrt(np.nanmean(res**2))

# ==========================================================
# Tendência
# ==========================================================
td_m_seculo = adt_coef * 365 * 100
td_mm_ano = adt_coef * 365 * 1000

print("Tendência média (m/século):", np.nanmean(td_m_seculo))
print("Desvio padrão da tendência:", np.nanstd(td_m_seculo))

# ==========================================================
# Função de plot
# ==========================================================
def plot_map(z, levels, titulo, textobarra, fname, cbar_ticks, clim=None, cmap='viridis'):
    z = z.T  # (lat, lon)

    fig, ax = plt.subplots(
        figsize=(12, 8),
        subplot_kw={'projection': ccrs.Mercator()}
    )

    ax.set_extent(
        [lon.min(), lon.max(), lat.min(), lat.max()],
        crs=ccrs.PlateCarree()
    )

    # Campo preenchido
    cf = ax.contourf(
        LON, LAT, z,
        levels=levels,
        cmap=cmap,
        extend='both',
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # Isolinhas com valores
    cs = ax.contour(
        LON, LAT, z,
        levels=levels[::2],
        colors='black',
        linewidths=1.0,
        transform=ccrs.PlateCarree(),
        zorder=5
    )
    ax.clabel(cs, fmt='%.2f', fontsize=10)

    # Continente correto (Natural Earth 10m)
    land = cfeature.NaturalEarthFeature(
        'physical', 'land', '10m',
        facecolor='0.85',
        edgecolor='black'
    )
    ax.add_feature(land, zorder=10)
    ax.coastlines(resolution='10m', linewidth=1.0, zorder=11)

    # Pontos
    if pontos is not None:
        xp, yp = zip(*pontos)
        ax.scatter(xp, yp, s=40, c='k',
                   transform=ccrs.PlateCarree(),
                   label='Pontos selecionados')
        ax.legend(loc='lower left')

    # Grade
    gl = ax.gridlines(
        draw_labels=True,
        linestyle='--',
        alpha=0.5
    )
    gl.top_labels = False
    gl.right_labels = False

    # Barra de cores
    cb = plt.colorbar(cf, ticks=cbar_ticks, shrink=0.8)
    cb.set_label(textobarra, fontsize=14)

    ax.set_title(titulo, fontsize=16)

    plt.savefig(fname, dpi=400, bbox_inches='tight')
    plt.close()

# ==========================================================
# Saída
# ==========================================================
os.makedirs("figs_estat", exist_ok=True)

# Pontos destacados
pontos = [
    (-45.950745, -1.025438),
    (-45.579382, -1.156578),
    (-45.254512, -1.485709),
    (-44.612493, -1.678496),
    (-44.216131, -2.334498),
    (-43.484629, -2.302975),
    (-42.735227, -2.502361),
    (-42.199938, -2.639296)
]

plot_map(
    adt_media,
    np.arange(0.50, 0.61, 0.005),
    "TDA (m) – média",
    "(m)",
    "figs_estat/adt_media.png",
    np.arange(0.50, 0.61, 0.02),
    clim=(0.50, 0.60)
)

plot_map(
    adt_despa,
    np.arange(0.045, 0.0651, 0.001),
    "TDA (m) – desvio padrão",
    "(m)",
    "figs_estat/adt_desvio_padrao.png",
    np.arange(0.045, 0.0651, 0.01),
    clim=(0.045, 0.065)
)

plot_map(
    adt_rmse,
    np.arange(0.035, 0.056, 0.001),
    "RMSE da regressão (m²)",
    "(m²)",
    "figs_estat/adt_rmse.png",
    np.arange(0.035, 0.056, 0.01),
    clim=(0.035, 0.055),
)

plot_map(
    td_m_seculo,
    np.arange(0.30, 0.41, 0.005),
    "Tendência da TDA (m/século)",
    "(m/século)",
    "figs_estat/adt_tendencia_m_seculo.png",
    np.arange(0.30, 0.41, 0.025),
    clim=(0.30, 0.40)
)

plot_map(
    td_mm_ano,
    np.arange(3.0, 4.01, 0.05),
    "Tendência da TDA (mm/ano)",
    "(mm/ano)",
    "figs_estat/adt_tendencia_mm_ano.png",
    np.arange(3.0, 4.01, 0.25),
    clim=(3.0, 4.0)
)
