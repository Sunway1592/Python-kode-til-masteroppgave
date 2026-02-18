# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:04:43 2024

@author: solve

Denne koden er laget for masteroppgaven til Solveig Pettersen.
"""

import os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import cv2
import statistics
import scipy
import sys
import pandas as pd



###############################################################################
'                    PANELINFORMASJON                                         '
###############################################################################

folder_name = '2' # Filnavn, slik at programmet kan hente ut informasjonen fra filsystemet
date = '02.10.2024' # Dato i format 'DD.MM.YYYY'
ppair = '2' # For titler, kun panelnr.
irradians = 391.96 #Kan være 'none' eller et tall


antall_celler = 288 # Alternativer: 288 for to panel, 144 for ett
exposure = 0.4 # eksponeringstid, f.eks 0.4 (ms)
irr_type = 'Direkte' # Alternativer: 'Direkte', 'Diffus'
side = 'Fremside' # Alternativer: 'Fremside', 'Bakside'
normalization_type = 'irradians' # Alternativer: 'irradians', 'median', 'gjennomsnitt', 'none'
plot_type = 'time development' # Alternativer: 'time development', 'exposure correction', 'front vs. back', 'direct vs. indirect', 'svart vs. blå', 'dPL-average', 'panel gjennomsnitt'


# Definer filnavn for lagring av data
#filnavn = f'panel_{ppair}_data_{normalization_type}.csv'

filnavn = f'{plot_type} - panel_{ppair}_data_{normalization_type}_{plot_type}.csv'
    
#Panelpar 2 [02.10.2024] (skala 500-850)


###############################################################################
'                         dPL-bilde                                           '
###############################################################################
print('\n //  Laster inn bilder og lager dPL-bilde:  //')


"Lager liste med bilder"
path = rf'C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Data Kjerringjordet\{date}\{folder_name}'
filelist = os.listdir(path)

images = []
for i in range(len(filelist)):
    bilde_navn = filelist[i]
    full_bilde_path = rf'{path}\{bilde_navn}'
    dette_bildet = skimage.io.imread(full_bilde_path).astype(np.int16)
    images.append(dette_bildet)


"Sjekker om det første bildet ble tatt i SC eller OC tilstanden"
plt.figure(figsize=(10, 6))
plt.plot(np.mean(images, axis=(1, 2))[:10], '-o')
#plt.title('Intensity of the 10 first images')
plt.xlabel('Bildenummer', fontsize=24, labelpad=20)
plt.ylabel('Intensitet', fontsize=24, labelpad=20)
plt.xticks(fontsize=22)  # Adjust to the size you want
plt.yticks(fontsize=22)  # Adjust to the size you want
plt.grid(False)
ax = plt.gca()          # get the current Axes object
ax.spines['top'].set_visible(False)   # Remove top and right frame edge
ax.spines['right'].set_visible(False)
plt.show()


"Skiller OC og SC bilder i to grupper"
images_oc = np.array([images[i] for i in range(1, np.shape(images)[0]-1, 2)])
images_sc = np.array([images[i] for i in range(0, np.shape(images)[0], 2)])


"Sjekker at signalene i de to gruppene har forskjellig intensitet."
#plt.figure()
#plt.plot(np.mean(images_oc, axis=(1, 2)), '-o')
#plt.plot(np.mean(images_sc, axis=(1, 2)), '-o')
#plt.title('Average intensity of all images')
#plt.xlabel('Image number')
#plt.ylabel('Intensity')
#plt.show()


"Beregner delta_PL bilde"
image_dPL = np.mean(images_oc, axis=0)-np.mean(images_sc, axis=0)
avr = np.mean(image_dPL[image_dPL > 50]) #Lager en midlertidig average for å lett kunne definere skala
print(f'Gjennomsnittlig dPL-verdi er ish {avr}')
clim=(0.85*avr, 1.3*avr)  # Etter hvert som jeg har jobbet med data så har jeg funnet ut at dette er typisk varians
#clim = (300, 860)
plt.figure()
plt.imshow(image_dPL, clim=clim, cmap='magma')
plt.axis('off')
plt.colorbar().ax.tick_params(labelsize=12)
#plt.title(f'Panelpar {folder_name} ({date})')
plt.show()


"Cropper bildet til minste rektangel"
coords = np.argwhere(image_dPL > 50)
y_min, x_min = coords.min(axis=0)
y_max, x_max = coords.max(axis=0)
image_cropped = image_dPL[y_min:y_max+1, x_min:x_max+1]
fig = plt.figure()
plt.imshow(image_cropped, clim=clim, cmap='magma')
plt.axis('off')
plt.colorbar().ax.tick_params(labelsize=12)
#plt.title(f'Panelpar {folder_name} ({date})')
#if irr_type == 'Diffus':
#    plt.title(f'Panelpar {folder_name} ({date}) [Diffus]')
plt.show()


"Lagrer bildet om ønskelig"
choice = input('dPL-bilde er klart.✅\n⏸ Ønsker du å lagre bildet?\nTrykk 1 for ja, ellers Enter: ').strip()
# Define destination path
base_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\dPL-bilder"
date_folder = os.path.join(base_path, date)
if choice == '1':
    # Create destination path only if it doesn't exist already
    os.makedirs(date_folder, exist_ok=True)
    # Use the folder name as the file name (number folder)
    png_file = f"Panelpar {folder_name} [{date}] (Skala {round(clim[0])} - {round(clim[1])}).png"
    if irr_type == 'Diffus':
        png_file = f"Panelpar {folder_name} (Skala {round(clim[0])} - {round(clim[1])} [Diffus]).png"
    file_path = os.path.join(date_folder, png_file)
    # Save image with title as metadata
    fig.savefig(file_path, bbox_inches='tight')
    print(f"💾 Lagret nytt dPL-bilde for dato ({date}):  '{png_file}'")






###############################################################################
"                       PREPROSESSERING                                       "
###############################################################################
print('\n //  Starter preprosessering:  //')


# KORRIGERER dPL-BILDET FOR ULIKE FORHOLD

"Flip image if it is a backside image"
if side == 'Bakside':
    image_cropped = np.fliplr(image_cropped)
    fig = plt.figure()
    plt.imshow(image_cropped, clim=clim, cmap='magma')
    plt.axis('off')
    plt.colorbar()
#    plt.title(f'Panelpar {folder_name} ({date}) [Bak]')
#    if irr_type == 'Diffus':
#        plt.title(f'Panelpar {folder_name} ({date}) [Bak, diffus irradians]')
    plt.show()
    print('Spegler bildet horisontalt.')


"Korriger for eksponering"
if exposure == 0.4:
    image_corrected = image_cropped
elif exposure in [0.2, 0.5, 0.9, 1.0, 1.68]:
    image_corrected = (image_cropped/exposure)*0.4  # Save image_cropped in case I need it later
    print('Korrigerer for eksponering.')
else:
    raise ValueError('Prøvde å korrigere for eksponering men eksponeringsverdien er ikke gyldig.')



# LAGER MASKEN SOM SKAL DEFINERE CELLER
print('Starter å lage masken.')

"Canny edge detection for å definere kanter"
# Gaussian Blur to reduce noise
blurred_image = cv2.GaussianBlur(image_cropped, (5, 5), 0)
# Define threshold for canne edge detection
low_threshold = 50 #50  # Lower threshold for edge detection
high_threshold = 100  #150 Higher threshold for edge detection
to_8bit = cv2.normalize(blurred_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
edges = cv2.Canny(to_8bit, low_threshold, high_threshold)
#Display the results
plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Canny Edge Detection')
plt.show()


"Closing Morphology"
# Create a kernel (structuring element) for the morphological operation
kernel = np.ones((5, 5), np.uint8)  # A 5x5 square kernel (you can adjust the size)
# Applying Closing Morphology to the thresholded image
closed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
inverted_mask = 255-closed_image # Have to invert the mask to get the cell areas to be "positive"
# Display the result
plt.figure(figsize=(6, 6))
plt.imshow(~inverted_mask, cmap='gray')
plt.axis('off')
plt.title('Mask after morphological Closing')
plt.show()


"Bruker label() for å definere regioner basert på masken og finner egenskaper med regionprops()"
labeled_regions, number_of_regions = skimage.measure.label(inverted_mask, connectivity=1, return_num=True)
region_properties = skimage.measure.regionprops(labeled_regions)
#print(f'Antall regioner før filtrering er {number_of_regions} og skal være lik {antall_celler}')


"Sjekk hvor stor et område normalt er"
# Sjekker pixelstørrelsen på de seks første områdene i tilfelle noen områder i starten er feildefinert
#for region in region_properties:
#    if 1 <= region.label <= 6:
#        print(f"Region {region.label}: Area = {region.area}")
# Dette brukes for å filtrere ut for store/små områder.


"Erase defined areas that are too small or too big"
# For panelene ute på jordet er det gjennomsnittlig  485 pixler per celle.
# For referansepanelet er det gjennomsnittlig 960 pixler per celle.
area = 485
if ppair == 'Referanse':
        area = 960
min_area = area/2  # Minimum area threshold. Half of one cell
max_area = area*2  # Maximum area threshold. Double of one cell
# Create a copy of the labeled_regions to modify
modified_labels = labeled_regions.copy()
# Iterate through each region and set labels outside the area thresholds to 0 (background)
for region in region_properties:
    area = region.area
    if area < min_area or area > max_area:
        # Set the label of the current region to zero
        coordinates = region.coords
        for coord in coordinates:
            inverted_mask[coord[0], coord[1]] = 0
plt.figure(figsize=(6, 6))
plt.imshow(~inverted_mask, cmap='gray')
plt.axis('off')
plt.title('Mask after filtering')
plt.show()


"Tar nye målinger av antall områder"
labeled_regions, number_of_regions = skimage.measure.label(inverted_mask, connectivity=1, return_num=True)
#if number_of_regions == antall_celler:
#    print(f'Antall cellelleområder etter filtrering er korrekt. ({number_of_regions} av {antall_celler})')
#else:
#    raise ValueError(f'Antall definerte celler er ikke korrekt. ({number_of_regions} av {antall_celler}). Manuell redigering kreves.')



"Manually remove areas if needed"
if number_of_regions != antall_celler:
    print(f"⏸❌ Forventet {antall_celler} celler, men registrerte {number_of_regions}.")
    to_remove_text = input("Inspiser labeled_regions. Skriv merkelappene som skal fjernes (separert med komma) og trykk enter for å fortsette: ")
    if to_remove_text.strip():
        # Parse into list of ints
        labels_to_remove = [int(x) for x in to_remove_text.split(",")]
        print("Fjerner merkelapper:", labels_to_remove)
        for lbl in labels_to_remove:
            inverted_mask[labeled_regions == lbl] = 0
        # Tar nye målinger av antall områder
        labeled_regions, number_of_regions = skimage.measure.label(inverted_mask, connectivity=1, return_num=True)
        if number_of_regions == antall_celler: # 144 for ett panel, 288 for et panelpar
            print(f'Antall definerte cellelleområder etter manuell redigering er korrekt ({number_of_regions} av {antall_celler}).')
        else:
            raise ValueError(f'Antall definerte celler er ikke korrekt. ({number_of_regions} av {antall_celler}). Sjekk kode for feil.')
else:
    input(f'Antall definerte celleområder er korrekt. ({number_of_regions} av {antall_celler})\n⏸ Inspiser masken visuelt. Trykk enter for å godkjenne:')


"""
# Option 1) Set areas to zero that were not properly defined
# Remember:
# -Slices are NOT inclusive of the ending number, so add 1.
# -Row slice first, then column slice second
#inverted_mask[473:474, 468:474] = 0 # Cell regions in the mask has value 255, edges has value 0

# Option 2) Delete an area that are not wanted by label
inverted_mask[labeled_regions == 1] = 0
inverted_mask[labeled_regions == 290] = 0
print('Områder har blitt manuelt fjernet')

#Plot
plt.figure(figsize=(6, 6))
plt.imshow(~inverted_mask, cmap='gray')
plt.axis('off')
plt.title('Mask after manual editing')
plt.show()


"Tar nye målinger av antall områder"
labeled_regions, number_of_regions = skimage.measure.label(inverted_mask, connectivity=1, return_num=True)
if number_of_regions == antall_celler: # 144 for ett panel, 288 for et panelpar
    print(f'Antall cellelleområder etter manuell redigering er korrekt. ({number_of_regions} av {antall_celler})')
else:
    raise ValueError(f'Antall definerte celler er ikke korrekt. ({number_of_regions} av {antall_celler}). Sjekk kode for feil.')

"""


#####################  Work this out before moving on  ########################

print('\nGir nye navn til merkelappene.')

"Renaming the labels to match the cell's position in the grid properly"
def relabel_regions(label_map, antall_celler, ppair):
    props = skimage.measure.regionprops(label_map)
    
    # Extract centroids (label, y, x)
    centroids = [(prop.label, prop.centroid[0], prop.centroid[1]) for prop in props]

    # Sort by y-coordinates first (approximate row detection)
    centroids.sort(key=lambda c: c[1])  

    # Cluster into rows based on y-coordinate proximity
    rows = []
    current_row = [centroids[0]]
    
    row_threshold = 10  # Adjust this based on spacing in your image
    for i in range(1, len(centroids)):
        _, y, _ = centroids[i]
        _, prev_y, _ = centroids[i - 1]

        if abs(y - prev_y) < row_threshold:
            current_row.append(centroids[i])
        else:
            rows.append(current_row)
            current_row = [centroids[i]]

    if current_row:
        rows.append(current_row)

    # Sort each row by x-coordinate
    for row in rows:
        row.sort(key=lambda c: c[2])
    new_label = 1
    # Create a mapping from old labels to new labels
    new_label_map = {}
    if antall_celler == 288:
        new_label = 1  # 1 or 145
    elif antall_celler == 144:
        if not ppair == 'Referanse':
            new_label = 145
    else:
        raise ValueError('Antall celler er ugyldig')

    for row in rows:
        for label, _, _ in row:
            new_label_map[label] = new_label
            new_label += 1

    # Create a new label map
    relabeled_map = np.zeros_like(label_map)
    for old_label, new_label in new_label_map.items():
        relabeled_map[label_map == old_label] = new_label

    return relabeled_map

new_labeled_image = relabel_regions(labeled_regions, antall_celler, ppair)


"Draw the areas to make sure all areas are defined correctly."
# Before
color_list = ['red', 'blue', 'green', 'yellow', 'purple']  # Use string color names
tegne_celler = skimage.color.label2rgb(labeled_regions, colors=color_list, bg_label=0, bg_color='black')
plt.figure()
plt.imshow(tegne_celler )
plt.axis('off')
plt.title('Definerte celler før sortering')
plt.show()

# After
color_list = ['red', 'blue', 'green', 'yellow', 'purple']  # Use string color names
tegne_celler = skimage.color.label2rgb(new_labeled_image, colors=color_list, bg_label=0, bg_color='black')
plt.figure()
plt.imshow(tegne_celler )
plt.axis('off')
plt.title('Definerte celler etter sortering')
plt.show()


"Test to make sure region 1 is to the left of region 2"
regions = skimage.measure.regionprops(new_labeled_image)
centroid1 = regions[0].centroid  # region 1
centroid2 = regions[1].centroid  # region 2
x1 = centroid1[1]
x2 = centroid2[1]
y1 = centroid1[0]
if not x1 < x2:
    raise ValueError(f"Region 1 is not to the left of Region 2: x1 = {x1}, x2 = {x2}")
if x1 >= 40 and y1 >= 40:
    raise ValueError("Region 1 is not in the top left corner")
    

input('⏸ Inspiser new_labeled_image. Trykk enter for å godkjenne:')

print('\nMasken er klar til bruk.')
print('Preprosessering ferdig.✅')





###############################################################################
"                          PROSESSERING                                       "
###############################################################################
print('\n //  Starter prosessering:  //')


"Gjennomsnittlig dPL-signal for panelparet"
average_dPL_value = np.mean(image_corrected[inverted_mask == 255])
print(f'Faktisk gjennomsnittlig dPL-signal for panelpar [{folder_name}] dato ({date}) er: {average_dPL_value}.')
# Påminnelse
if not exposure == 0.4:
    print('Husk at dPL-bildet har blitt korrigert for eksponering.')


"Normalize the data based on chosen method"
if normalization_type == 'gjennomsnitt':
    gjennomsnitt = np.mean(image_corrected[new_labeled_image > 0])
    norm_gjsn= image_corrected / gjennomsnitt
    normalisert = norm_gjsn
elif normalization_type == 'irradians':
    norm_irr = image_corrected / irradians
    normalisert = norm_irr
elif normalization_type == 'median':
    # Ta ut alle områder i bildet som ikke er i bakgrunnen
    valid_pixels = image_corrected[new_labeled_image > 0]
    # Regn ut medianen for alle pixler som ikke er bakgrunn
    median = np.median(valid_pixels)
    norm_median = image_corrected / median
    normalisert = norm_median
elif normalization_type == 'none':
    normalisert = image_corrected
else:
    raise ValueError('Normaliseringsmetode ikke gjenkjent')
print(f'dPL-bildet er blitt normalisert med {normalization_type}.')


#Plot normalisert bilde
plt.figure()
clim_norm = (1.0, 1.7)
plt.imshow(normalisert, clim=clim_norm, cmap='magma')
plt.axis('off')
plt.colorbar().ax.tick_params(labelsize=12)
#plt.title(f'Panelpar {folder_name} i {irr_type} irradians ({date}) normalisert med {normalization_type}')
plt.show()





"Finn gjennomsnittsverdi for hver celle"
props = skimage.measure.regionprops(new_labeled_image, normalisert)
mean_values = [(prop.label, prop.intensity_mean) for prop in props]
#print(mean_values)
print('Gjennomsnittlig dPL-signal for hver celle er beregnet.')


"Finn gjennomsnittlig normalisert dPL-signal for hvert panel"
# For det svarte panelet
if ppair == 'Referanse':
    top_mask = (new_labeled_image >= 1) & (new_labeled_image <= 144)
    top_panel_mean = np.mean(normalisert[top_mask])
    print(f'Gjennomsnittlig dP-signal for referansepanelet normalisert med {normalization_type} er {top_panel_mean}')
else:
    top_mask = (new_labeled_image >= 1) & (new_labeled_image <= 144)
    top_panel_mean = np.mean(normalisert[top_mask])
    print(f'Gjennomsnittlig dP-signal for øverste panel normalisert med {normalization_type} er {top_panel_mean}')
    # For det blå panelet
    bottom_mask  = (new_labeled_image >= 145) & (new_labeled_image <= 288)
    bottom_panel_mean  = np.mean(normalisert[bottom_mask])
    print(f'Gjennomsnittlig dP-signal for nederste panel normalisert med {normalization_type} er {bottom_panel_mean}')


print('Prosessering fullført.✅')
input('\n⏸ Trykk Enter for å fortsette til lagring:')





# SAFETY NET
#raise SystemExit('Ready to save?')




###############################################################################
"                        SAVE AND DELETE DATA                                 "
###############################################################################



"Definer mappe for lagring"
if plot_type == 'time development':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Data tidsprogresjon"
elif plot_type == 'exposure correction':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Eksponeringstid"
elif plot_type == 'front vs. back':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Bakside vs. fremside"
elif plot_type == 'direct vs. indirect':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Direkte vs. diffus"
elif plot_type == 'svart vs. blå':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Svart vs. blå"
elif plot_type =='panel gjennomsnitt':
    folder_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Data tidsprogresjon"
else:
    raise ValueError('Plot-type ikke gjenkjent.')
csv_path = os.path.join(folder_path, filnavn)


"Make a datset when I am happy with it"
# Convert to DataFrame
if not plot_type == 'panel gjennomsnitt':
    df = pd.DataFrame(mean_values, columns=["cell", "mean_signal"])
    
if plot_type == 'time development':
    df["date"] = date  # Add the variable column (date, exposure, side etc..)
elif plot_type == 'exposure correction':
    df['exposure'] = exposure
elif plot_type == 'front vs. back':
    df['side'] = side
elif plot_type == 'direct vs. indirect':
    df['irr type'] = irr_type
elif plot_type == 'svart vs. blå':
    df['date'] = date
    df['top panel mean'] = top_panel_mean
    df['bottom panel mean'] = bottom_panel_mean
elif plot_type == 'panel gjennomsnitt':
    df = pd.DataFrame({'top panel mean': [top_panel_mean], 'bottom panel mean': [bottom_panel_mean], 'date': [date]})


"Define referance point data frame"
#date_referance = '25.09.2024'
#avr_dPL_value_referance = 2.1979598
#df_ref = pd.DataFrame({'black panel mean': [avr_dPL_value_referance], 'blue panel mean': [avr_dPL_value_referance], 'date': [date_referance]})


"Save dataset as new file or add to preexisting csv-file"
try:
    # If file exists, load and append
    existing = pd.read_csv(csv_path)
    combined = pd.concat([existing, df], ignore_index=True)
    combined.to_csv(csv_path, index=False)
    print(f'\n💾 Saved to existing file: {filnavn}')
except FileNotFoundError:
    # If file doesn't exist yet, create it
    df.to_csv(csv_path, index=False)
    print(f'\n🆕💾 Created new file: {filnavn}')




"If needed: Delete wrong data"
#df = pd.read_csv(csv_path)
# Remove rows where date is the one you are working with (or choose any other bad date)
#df = df[df["date"] != '08.03.2025']
#df = df[df["cell"] >= 144]
#df.to_csv(csv_path, index=False)




input('\n⏸ Trykk Enter for å se plottet:')




###############################################################################
"                              DATA PLOTTING                                  "
###############################################################################
# Load full dataset by running the path definition bulk written above the save part


# TIME DEVELOPMENT
if plot_type == 'time development':
    # Make sure the file name and directory is defined before reading the dataset.
    df = pd.read_csv(csv_path)
    # Parse dates in DD.MM.YYYY format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    df = df.sort_values(by=["cell", "date"])

    # FOR TOP PANEL
    plt.figure(figsize=(16, 8))
    for cell_id, group in df[df['cell'] <= 144].groupby("cell"):
        color = "red"
        plt.plot(group["date"], group["mean_signal"], marker = 'o', color=color, linewidth=0.8, alpha=0.6)
    # Modify the graph
    #plt.title(f"dPL-signal normalisert med {normalization_type} (Panel {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=18)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=18)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["mean_signal"].min() * 0.9, df["mean_signal"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=14)  # Adjust to the size you want
    plt.yticks(fontsize=14)  # Adjust to the size you want
    plt.grid(False)
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(title="Celle i panel", fontsize=14, title_fontsize=16)
    plt.show()

    # FOR BOTTOM PANEL
    plt.figure(figsize=(16, 8))
    for cell_id, group in df[df['cell'] > 144].groupby("cell"):
        color = "blue"
        plt.plot(group["date"], group["mean_signal"], marker = 'o', color=color, linewidth=0.8, alpha=0.6)
    # Modify the graph
    #plt.title(f"dPL-signal normalisert med {normalization_type} (Panel {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=18)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=18)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["mean_signal"].min() * 0.9, df["mean_signal"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=14)  # Adjust to the size you want
    plt.yticks(fontsize=14)  # Adjust to the size you want
    plt.grid(False)
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(title="Celle i panel", fontsize=14, title_fontsize=16)
    plt.show()

    # FOR THE AVERAGE VALUES
    # Parse dates in DD.MM.YYYY format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    df = df.sort_values(by=["cell", "date"])
    # Plot one line per cell
    plt.figure(figsize=(16, 8))
    # Plot gjennomsnitt for hvert panel i svarte diamanter
    top_df = df[["date", "top panel mean"]].dropna(subset=["top panel mean"]).drop_duplicates("date").sort_values("date")
    bottom_df  = df[["date", "bottom panel mean"]].dropna(subset=["bottom panel mean"]).drop_duplicates("date").sort_values("date")
    plt.plot(top_df["date"], top_df["top panel mean"], color="red",  lw=3, marker="D", markersize=8, label="Gjennomsnitt svart panel")
    plt.plot(bottom_df["date"],  bottom_df["bottom panel mean"], color="blue", lw=3, marker="D", markersize=8, label="Gjennomsnitt blått panel")
    # Modify the graph
    #plt.title(f"Gjennomsnittlig dPL-signal for svarte vs. blå panel (panelpar {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=18, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=18, labelpad=20)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["bottom panel mean"].min() * 0.9, df["bottom panel mean"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=14)  # Adjust to the size you want
    plt.yticks(fontsize=14)  # Adjust to the size you want
    plt.grid(False)
    plt.legend(title="Gjennomsnittlig dPL-verdi", fontsize=14, title_fontsize=16)
    plt.show()

if plot_type == 'svart vs. blå':
    csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Svart vs. blå\panel_12_data_irradians.csv"
    # Make sure the file name and directory is defined before reading the dataset.
    df = pd.read_csv(csv_path)
    # Parse dates in DD.MM.YYYY format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    df = df.sort_values(by=["cell", "date"])

    # FOR TOP PANEL
    plt.figure(figsize=(16, 8))
    for cell_id, group in df[df['cell'] <= 144].groupby("cell"):
        color = "black"
        plt.plot(group["date"], group["mean_signal"], marker = 'o', color=color, linewidth=0.8, alpha=0.6)
    # Modify the graph
    #plt.title(f"dPL-signal normalisert med {normalization_type} (Panel {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=24, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=24, labelpad = 20)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["mean_signal"].min() * 0.9, df["mean_signal"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=22)  # Adjust to the size you want
    plt.yticks(fontsize=22)  # Adjust to the size you want
    plt.grid(False)
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=20)
    plt.show()

    # FOR BOTTOM PANEL
    plt.figure(figsize=(16, 8))
    for cell_id, group in df[df['cell'] > 144].groupby("cell"):
        color = "blue"
        plt.plot(group["date"], group["mean_signal"], marker = 'o', color=color, linewidth=0.8, alpha=0.6)
    # Modify the graph
    #plt.title(f"dPL-signal normalisert med {normalization_type} (Panel {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=24, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=24, labelpad=20)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["mean_signal"].min() * 0.9, df["mean_signal"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=22)  # Adjust to the size you want
    plt.yticks(fontsize=22)  # Adjust to the size you want
    plt.grid(False)
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=20)
    plt.show()


    csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Svart vs. blå\panel_12_data_irradians - panelgjennomsnitt med referanse.csv"
    df = pd.read_csv(csv_path)
    # FOR THE AVERAGE VALUES
    # Parse dates in DD.MM.YYYY format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')
    #df = df.sort_values(by=["cell", "date"])
    df = df.sort_values(by=["date"])
    # Plot one line per cell
    plt.figure(figsize=(16, 8))
    # Plot gjennomsnitt for hvert panel i svarte diamanter
    top_df = df[["date", "black panel mean"]].dropna(subset=["black panel mean"]).drop_duplicates("date").sort_values("date")
    bottom_df  = df[["date", "blue panel mean"]].dropna(subset=["blue panel mean"]).drop_duplicates("date").sort_values("date")
    plt.plot(top_df["date"], top_df["black panel mean"], color="black",  lw=3, marker="D", markersize=8, label="Gjennomsnitt svart panel")
    plt.plot(bottom_df["date"],  bottom_df["blue panel mean"], color="blue", lw=3, marker="D", markersize=8, label="Gjennomsnitt blått panel")
    # Add a black star to the first datapoint
    plt.scatter(df["date"].iloc[0],df["black panel mean"].iloc[0], marker="*", s=400, color="black", linewidth=0.8, zorder=5, label="Referanseverdi")
    # Modify the graph
    #plt.title(f"Gjennomsnittlig dPL-signal for svarte vs. blå panel (panelpar {ppair})", fontsize=22)
    plt.xlabel("Dato", fontsize=24, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=24, labelpad=20)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(df["blue panel mean"].min() * 0.9, df["black panel mean"].max() * 1.05)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=22)  # Adjust to the size you want
    plt.yticks(fontsize=22)  # Adjust to the size you want
    plt.grid(False)
    plt.legend(fontsize=20)
    plt.show()


# EXPOSURE
elif plot_type == 'exposure correction':
    # Make sure the file name and directory is defined before reading the dataset.
    #csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Eksponeringstid\Referansepanel (26.02.2025) - Eksponeringstid.csv"
    csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Eksponeringstid\Referansepanel (26.02.2025) - Eksponeringstid.csv"
    
    df = pd.read_csv(csv_path)
    # Sort values by exposure, then by cell.
    df = df.sort_values(by=["exposure", "cell"])
    # Let's plot, and add different colors for different exposure times
    plt.figure(figsize=(16, 8))
    for exposure_time, group in df.groupby("exposure"):
        if exposure_time == 0.4:
            color = 'red'
        elif exposure_time == 0.5:
            color = 'blue'
        elif exposure_time == 0.9:
            color = 'green'
        elif exposure_time == 1.0:
            color = 'orange'
        elif exposure_time == 1.68:
            color = 'purple'
        else:
            raise ValueError('Exposure not defined')
        plt.plot(group["cell"], group["mean_signal"], marker = 'o', color=color, linestyle = 'None', alpha=0.6, label=exposure_time)
    # Modify the graph
    #plt.title(f'dPL-signal til referansepanelet ({date}) ved ulike eksponerignstider', fontsize=22)
    plt.xlabel("Celle", fontsize=24, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=24, labelpad=20)
    plt.xticks(fontsize=22)  # Adjust 12 to the size you want
    plt.yticks(fontsize=22)  # Adjust 12 to the size you want
    #plt.xlim(start_date, end_date)  # x-axis limits
    plt.ylim(0, df["mean_signal"].max() * 1.4)  # y-axis limits from 0 to max + margin
    plt.grid(False)
    plt.tight_layout()
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(title='Eksponeringstid [ms]', title_fontsize=20, fontsize=20, ncol=2)
    plt.show()


# FRONT vs BACK
elif plot_type == 'front vs. back':
    # Make sure the file name and directory is defined before reading the dataset.
    #csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Bakside vs. fremside\12 exp 0.4 (11.02.2025) - Bakside vs fremside.csv"
    df = pd.read_csv(csv_path)
    # Sort values by exposure, then by cell.
    df = df.sort_values(by=["side", "cell"])
    # Let's plot, and add different colors for different exposure times
    plt.figure(figsize=(16, 8))
    for side, group in df.groupby("side"):
        if side == 'Fremside':
            color = 'red'
        elif side == 'Bakside':
            color = 'blue'
        else:
            raise ValueError('Side of panel not defined')
        plt.plot(group["cell"], group["mean_signal"], marker = 'o', color=color, linestyle = 'None', alpha=0.6, label=side)
    # Modify the graph
    #plt.title(f'dPL-signal til panelpar {ppair} fra fremside og bakside ({date})', fontsize=22)
    plt.xlabel("Celle", fontsize=22, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=22, labelpad=20)
    plt.xticks(fontsize=22)  # Adjust to the size you want
    plt.yticks(fontsize=22)  # Adjust to the size you want
    plt.xlim(0, 290)  # x-axis limits
    plt.ylim(0, df["mean_signal"].max() * 1.1)  # y-axis limits from 0 to max + margin
    plt.legend(fontsize=18)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


# DIRECT vs INDIRECT IRRADIANCE
elif plot_type == 'direct vs. indirect':
    # Make sure the file name and directory is defined before reading the dataset.
    #csv_path = r"C:\Users\solve\OneDrive - Norwegian University of Life Sciences\Skole\MILF 24-25\Masteroppgave\Dataprossessering\Direkte vs. diffus\2 - Direkte vs. diffus (14.03.2025) og (08.03.2025).csv"
    df = pd.read_csv(csv_path)
    # Sort values by exposure, then by cell.
    df = df.sort_values(by=["irr type", "cell"])
    # Let's plot, and add different colors for different exposure times
    plt.figure(figsize=(16, 8))
    for irr_type, group in df.groupby("irr type"):
        if irr_type == 'Direkte':
            color = 'red'
        elif irr_type == 'Diffus':
            color = 'blue'
        else:
            raise ValueError('Irradiance type not defined')
        plt.plot(group["cell"], group["mean_signal"], marker = 'o', color=color, linestyle = 'None', alpha=0.6, label=irr_type)
    # Modify the graph
    #plt.title(f'Normalisert dPL-signal til panelpar {ppair} i direkte og diffus irradians', fontsize=22)
    plt.xlabel("Celle", fontsize=24, labelpad=20)
    plt.ylabel("Gjennomsnittlig dPL-signal per celle", fontsize=24, labelpad=20)
    plt.xticks(fontsize=22)  # Adjust 12 to the size you want
    plt.yticks(fontsize=22)  # Adjust 12 to the size you want
    plt.xlim(0, 290)  # x-axis limits
    plt.ylim(1, df["mean_signal"].max() * 1.1)  # y-axis limits from 0 to max + margin
    plt.legend(fontsize=20)
    plt.grid(False)
    plt.tight_layout()
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.show()


# PANEL MEAN VALUES
elif plot_type == 'panel gjennomsnitt':
    # Read the file
    df = pd.read_csv(csv_path)

    # Parse dates in DD.MM.YYYY format
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors='coerce')

    # Sort values
    df = df.sort_values(by=["date"])

    # Colour map for panels
    color_map = {1: "black", 2: "blue"}

    # Plot
    plt.figure(figsize=(16, 8))
    plt.plot(df["date"], df["top panel mean"], lw=3, marker="D", color='red', markersize=8, label='Øverste panel')
    plt.plot(df["date"], df["bottom panel mean"], lw=3, marker="D", color='blue', markersize=8, label='Laveste panel')
    # Overlay a different marker for the reference (first row)
    plt.scatter(df["date"].iloc[0], df["top panel mean"].iloc[0], color='black', marker='*', s=400, label="Referanseverdi", zorder=5)    
    # Modify the graph
    plt.xlabel("Dato", fontsize=24, labelpad=15)
    plt.ylabel("Gjennomsnittlig dPL-signal per panel", fontsize=24, labelpad=15)
    #plt.xlim(start_date, end_date)  # Set x-axis limits
    plt.ylim(1, 2.4)  # y-axis limits from 0 to max + margin
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xticks(fontsize=22)  # Adjust to the size you want
    plt.yticks(fontsize=22)  # Adjust to the size you want
    plt.grid(False)
    ax = plt.gca()          # get the current Axes object
    ax.spines['top'].set_visible(False)   # Remove top and right frame edge
    ax.spines['right'].set_visible(False)
    plt.legend(fontsize=18)
    plt.show()
        
