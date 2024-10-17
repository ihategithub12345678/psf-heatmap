import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fwhm_extractor import fwhm_extractor

beads_list = fwhm_extractor()

wavelength = 590
numerical_aperture = 1.4
theo_xy = 0.51 * (wavelength / 1000) / numerical_aperture
theo_z = 2 * (wavelength / 1000) / numerical_aperture ** 2

complete_beads = []
incomplete_beads = []

for bead in beads_list:
    if all(key in bead for key in ['pos_x', 'pos_y', 'fwhm_x', 'fwhm_y', 'fwhm_z']):
        complete_beads.append(bead)
    else:
        incomplete_beads.append(bead)

x = [bead['pos_x'] for bead in complete_beads]
y = [bead['pos_y'] for bead in complete_beads]
z_x = [bead['fwhm_x'] / theo_xy for bead in complete_beads]
z_y = [bead['fwhm_y'] / theo_xy for bead in complete_beads]
z_z = [bead['fwhm_z'] / theo_z for bead in complete_beads]

x_incomplete = [bead.get('pos_x', np.nan) for bead in incomplete_beads]
y_incomplete = [bead.get('pos_y', np.nan) for bead in incomplete_beads]

xi = np.linspace(min(x), max(x), 100)
yi = np.linspace(min(y), max(y), 100)
xi, yi = np.meshgrid(xi, yi)

zi_x = griddata((x, y), z_x, (xi, yi), method='cubic')
zi_y = griddata((x, y), z_y, (xi, yi), method='cubic')
zi_z = griddata((x, y), z_z, (xi, yi), method='cubic')

zi_x_linear = griddata((x, y), z_x, (xi, yi), method='linear')
zi_y_linear = griddata((x, y), z_y, (xi, yi), method='linear')
zi_z_linear = griddata((x, y), z_z, (xi, yi), method='linear')

error_x = np.abs(zi_x - zi_x_linear)
error_y = np.abs(zi_y - zi_y_linear)
error_z = np.abs(zi_z - zi_z_linear)

fig, axs = plt.subplots(2, 3, figsize=(18, 12))

def plot_heatmap(ax, zi, title, is_error=False):
    im = ax.imshow(zi, extent=[min(x), max(x), max(y), min(y)], origin='upper', cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    if not is_error:
        ax.scatter(x, y, c='red', s=10, label='Complete data')
        ax.scatter(x_incomplete, y_incomplete, c='gray', s=10, alpha=0.5, label='Incomplete data')
        ax.legend()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

# Plot heatmaps
plot_heatmap(axs[0, 0], zi_x, 'X FWHM / Theoretical')
plot_heatmap(axs[0, 1], zi_y, 'Y FWHM / Theoretical')
plot_heatmap(axs[0, 2], zi_z, 'Z FWHM / Theoretical')

# Plot error heatmaps
plot_heatmap(axs[1, 0], error_x, 'X Interpolation Error', True)
plot_heatmap(axs[1, 1], error_y, 'Y Interpolation Error', True)
plot_heatmap(axs[1, 2], error_z, 'Z Interpolation Error', True)

plt.tight_layout(pad=10)
plt.show()
