import re
import os
import csv
from scipy.interpolate import griddata
from pykrige.ok import OrdinaryKriging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

images_directory = "GreenBeadsIMG4R"

cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["darkblue", "lightblue", "lightgreen", "orange", "red"])

vmin, vmax = 0, 2

def process_file(file_path):
    result = {
        "Mes./theory ratio": {},
        "Fit Goodness": {},
        "Lateral Asymmetry": None,
        "Bead coordinates (pixels)": {}
    }

    with open(file_path, 'r') as file:
        data = file.read().split('\n')
        resolution_table_start = False
        lateral_asymmetry_table_start = False

        for line in data:
            if "Bead original coordinates(in pixels)" in line:
                coords = line.split('\t')[2].strip()
                if coords:
                    try:
                        x, y = map(float, coords.split(', '))
                        result["Bead coordinates (pixels)"]["X"] = x
                        result["Bead coordinates (pixels)"]["Y"] = y
                    except ValueError:
                        print(f"Invalid coordinates in {file_path}")

            elif "Resolution" in line:
                resolution_table_start = True
            elif resolution_table_start and line.strip() and line.split("\t")[0] != "Channel":
                parts = line.split('\t')
                if len(parts) >= 7:
                    dimension, fit_goodness, ratio = parts[2], parts[5], parts[6]
                    if is_numeric(ratio):
                        ratio_value = float(ratio)
                        result["Fit Goodness"][dimension] = fit_goodness
                        result["Mes./theory ratio"][dimension] = ratio_value

            elif "Lateral Asymmetry" in line:
                lateral_asymmetry_table_start = True
            elif lateral_asymmetry_table_start and line.strip().startswith("Channel 0") and line.strip():
                parts = line.split('\t')
                if len(parts) >= 2 and is_numeric(parts[1]):
                    result["Lateral Asymmetry"] = float(parts[1])
                break

    return result

def get_bead_data(image_folders):
    beads_list = []
    for image in image_folders:
        image_path = os.path.join(images_directory, image)
        bead_folders = [item for item in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, item)) and re.match(r'^bead\d+$', item)]

        for bead in bead_folders:
            data_folder = f"{images_directory}_{image}_{bead}_data"
            file_path = os.path.join(image_path, bead, data_folder, f"{images_directory}_{image}_{bead}_results.xls")

            try:
                result = process_file(file_path)
                result["Image"] = image
                result["Bead"] = bead
                beads_list.append(result)
            except FileNotFoundError:
                print(f"\nError: {file_path} not found")
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")

    return beads_list

image_folders = [item for item in os.listdir(images_directory) if os.path.isdir(os.path.join(images_directory, item)) and re.match(r'^Image \d+$', item)]
beads_list = get_bead_data(image_folders)

# CSV writing
csv_filename = f"{images_directory}_bead_data.csv"
fieldnames = ['Image', 'Bead', 'Mes./theory ratio X', 'Mes./theory ratio Y', 'Mes./theory ratio Z',
              'Lateral Asymmetry', 'Bead coordinates X (pixels)', 'Bead coordinates Y (pixels)']

with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in beads_list:
        writer.writerow({
            'Image': item['Image'],
            'Bead': item['Bead'],
            'Mes./theory ratio X': item['Mes./theory ratio'].get('X', ''),
            'Mes./theory ratio Y': item['Mes./theory ratio'].get('Y', ''),
            'Mes./theory ratio Z': item['Mes./theory ratio'].get('Z', ''),
            'Lateral Asymmetry': item['Lateral Asymmetry'],
            'Bead coordinates X (pixels)': item['Bead coordinates (pixels)']['X'],
            'Bead coordinates Y (pixels)': item['Bead coordinates (pixels)']['Y']
        })

print(f"CSV file '{csv_filename}' has been created successfully.")

# Kriging Interpolation
variogram_model = "gaussian"

def is_valid_dimension(bead, dimension):
    if dimension in ['X', 'Y', 'Z']:
        return (dimension in bead["Mes./theory ratio"] and
                dimension in bead["Fit Goodness"] and
                float(bead["Fit Goodness"][dimension]) >= 0.9)
    elif dimension == 'Lateral':
        return bead["Lateral Asymmetry"] is not None

def get_valid_beads(beads_list, dimension):
    return [bead for bead in beads_list
            if ("X" in bead["Bead coordinates (pixels)"] and
                "Y" in bead["Bead coordinates (pixels)"] and
                is_valid_dimension(bead, dimension))]

def extract_data(beads, dimension):
    x = np.array([bead["Bead coordinates (pixels)"]["X"] for bead in beads])
    y = np.array([bead["Bead coordinates (pixels)"]["Y"] for bead in beads])
    z = np.array([bead["Lateral Asymmetry"] if dimension == 'Lateral' else bead["Mes./theory ratio"][dimension] for bead in beads])
    return x, y, z

dimensions = ['X', 'Y', 'Z', 'Lateral']
valid_beads = {dim: get_valid_beads(beads_list, dim) for dim in dimensions}
extracted_data = {dim: extract_data(valid_beads[dim], dim) for dim in dimensions}

x_min = min(np.min(data[0]) for data in extracted_data.values())
x_max = max(np.max(data[0]) for data in extracted_data.values())
y_min = min(np.min(data[1]) for data in extracted_data.values())
y_max = max(np.max(data[1]) for data in extracted_data.values())

xi, yi = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

def kriging_interpolation(x, y, z_values, xi, yi):
    OK = OrdinaryKriging(x, y, z_values, variogram_model='linear', verbose=False, enable_plotting=False)
    zi, _ = OK.execute('grid', xi, yi)
    return zi

interpolated_data = {dim: kriging_interpolation(*extracted_data[dim], xi, yi) for dim in dimensions}

def plot_heatmap(ax, zi, title, z_values, x, y):
    im = ax.imshow(zi, extent=[x_min, x_max, y_max, y_min], origin='upper', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.scatter(x, y, c='red', s=10, label='Complete data')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

fig, axs = plt.subplots(2, 2, figsize=(20, 20))
titles = ['X FWHM / Theoretical (Kriging)', 'Y FWHM / Theoretical (Kriging)',
          'Z FWHM / Theoretical (Kriging)', 'Lateral Asymmetry (Kriging)']

for i, (dim, title) in enumerate(zip(dimensions, titles)):
    plot_heatmap(axs[i//2, i%2], interpolated_data[dim], title, *extracted_data[dim])

plt.tight_layout()
plt.show()
