import re
import os

def fwhm_extractor(directory_path='sample-data', proj_name="redbead_summary"):
    items = os.listdir(directory_path)
    bead_folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item)) and re.match(r'^bead\d+$', item)]

    beads_list = []

    for folder in bead_folders:
        file_path = os.path.join(directory_path, folder)
        bead_dict = {'name': folder}

        try:
            with open(os.path.join(file_path, proj_name + ".xls"), 'r') as file:
                for line in file:
                    if "Bead coordinates in original image" in line:
                        coords = line.split('\t')[1].split(',')
                        bead_dict['pos_x'] = float(coords[0])
                        bead_dict['pos_y'] = float(coords[1])
                    match = re.search(r'\t([XYZ])\t([\d.]+)', line)
                    if match:
                        dimension, value = match.groups()
                        bead_dict[f'fwhm_{dimension.lower()}'] = float(value)

            beads_list.append(bead_dict)
        except FileNotFoundError:
            print(f"\nError: {proj_name}.xls not found in {folder}")
        except Exception as e:
            print(f"\nError processing {folder}: {str(e)}")

    return beads_list
