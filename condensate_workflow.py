import numpy as np
import pandas as pd
import cv2
import json
import os
import matplotlib.pyplot as plt     
import re

"""
adaptive_threshold.py

A pipeline for analyzing microscopy images of biomolecular condensates using adaptive 
thresholding to segment droplets and quantify their properties.

This module provides functions to:
- Perform adaptive thresholding and contour extraction
- Process multiple image folders with configurable parameters
- Calculate droplet area, condensed fraction, and circularity
- Generate visualizations and export data for statistical analysis

The script expects images organized into folders by experimental condition (e.g., concentration)
and uses a config.json file to specify thresholding parameters for each folder.
"""


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def resize_for_display(image, max_width=640, max_height=640):
    """
    Resize an image for display while preserving aspect ratio.
    
    Args:
        image (numpy.ndarray): Input image
        max_width (int): Maximum width in pixels (default: 640)
        max_height (int): Maximum height in pixels (default: 640)
    
    Returns:
        numpy.ndarray: Resized image, or original if within size limits
    """
    h, w = image.shape[:2]
    if h > max_height or w > max_width:
        scale = min(max_height/h, max_width/w)
        new_size = (int(w*scale), int(h*scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image


def calculate_circularity(contour):
    """
    Calculate the circularity metric for a contour.
    
    Circularity is defined as (4π × Area) / (Perimeter²)
    - Perfect circle: 1.0
    - Elongated shapes: approaches 0
    
    Args:
        contour (numpy.ndarray): OpenCV contour object
    
    Returns:
        float: Circularity value between 0 and 1
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return 0
    
    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return min(circularity, 1.0)


def get_sorted_concentration(folder_path):
    """
    Extract numerical concentration values from subfolder names.
    
    Attempts to parse folder names as:
    1. Pure numbers (e.g., "5.0")
    2. Numbers with "uM" suffix (e.g., "5.0uM")
    3. Numbers at the end of the name (e.g., "sample_5.0")
    
    Args:
        folder_path (str): Path to directory containing condition subfolders
    
    Returns:
        list: Sorted list of concentration values
    """
    concentrations = []
    for folder_name in os.listdir(folder_path):
        print(folder_name)
        if os.path.isdir(os.path.join(folder_path, folder_name)) and (not folder_name.startswith('.')):
            try:
                concentration = float(folder_name)
                concentrations.append(concentration)
            except ValueError:
                try:
                    concentration = re.search(r'(\d*\.?\d+)uM', folder_name, re.IGNORECASE)
                    if concentration:
                        concentrations.append(float(concentration.group(1)))
                    else:
                        concentration_end = re.search(r'(\d*\.?\d+)$', folder_name)
                        if concentration_end:
                            concentrations.append(float(concentration_end.group(1)))
                except:
                    continue
    return sorted(concentrations)


# ==============================================================================
# IMAGE PROCESSING FUNCTIONS
# ==============================================================================

def binarize(original_image, contrast=-1, block_size=51, excludeSmallDots=0, display=False, channel=None, calculate_circularity=False):
    """
    Convert a grayscale image to a binary mask using adaptive thresholding.
    
    This function applies adaptive thresholding with edge padding to handle boundary effects,
    identifies contours, filters small objects, and optionally calculates circularity metrics.
    
    Args:
        original_image (numpy.ndarray): Input image (grayscale or multi-channel)
        contrast (int): Constant subtracted from the adaptive threshold mean (default: -1)
        block_size (int): Size of pixel neighborhood for threshold calculation (must be odd, default: 51)
        excludeSmallDots (int): Minimum contour area in pixels to retain (default: 0)
        display (bool): Whether to display the results using OpenCV windows (default: False)
        channel (int): Channel index to use for multi-channel images (default: None, uses channel 0)
        calculate_circularity (bool): Whether to calculate circularity for each contour (default: False)
    
    Returns:
        tuple: Contains the following elements:
            - original_image (numpy.ndarray): Input image (unchanged)
            - contour_img (numpy.ndarray): Image with contours drawn
            - final_binary (numpy.ndarray): Binary mask of retained contours
            - circularities (list, optional): List of circularity values if calculate_circularity=True
    """
    
    if len(original_image.shape) == 3:
        num_channels = original_image.shape[2]

        if channel is not None:
            if channel < 0 or channel >= num_channels:
                raise ValueError(f"Invalid channel index {channel}. Image has {num_channels} channels.")
            original_image = original_image[:, :, channel]
            print(f"Using channel {channel} for binarization.")
        else:
            original_image = original_image[:, :, 0]

    # Apply adaptive thresholding with reflective border padding
    padded_img = cv2.copyMakeBorder(original_image, 20, 20, 20, 20, cv2.BORDER_REFLECT)
    binary_image = cv2.adaptiveThreshold(
        padded_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, block_size, contrast)
    
    binary_image = binary_image[20:-20, 20:-20]

    contour_img = original_image.copy()
    final_binary = np.zeros_like(binary_image)
    circularities = [] if calculate_circularity else None
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cntr in contours:
        area = cv2.contourArea(cntr)
        
        if area <= excludeSmallDots:
            continue

        if calculate_circularity:
            circularity = calculate_circularity(cntr)
            circularities.append(circularity)
        
        cv2.drawContours(contour_img, [cntr], 0, (255, 105, 65), 2)
        cv2.drawContours(final_binary, [cntr], 0, 255, -1)
        
    if display:
        contour_img = resize_for_display(contour_img, max_width=640, max_height=640)
        cv2.imshow("Final Binary Image", final_binary)
        cv2.imshow("Contour Image", contour_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if calculate_circularity:
        return original_image, contour_img, final_binary, circularities
    else:
        return original_image, contour_img, final_binary


def process_file(image_file, folder_path, params, display=False, channel=None, calculate_circularity=False):
    """
    Process a single image file to extract droplet properties.
    
    Performs binarization, connected component labeling, and calculates droplet areas
    and condensed fraction.
    
    Args:
        image_file (str): Filename of the image to process
        folder_path (str): Path to the folder containing the image
        params (dict): Dictionary containing 'contrast' and 'block_size' parameters
        display (bool): Display processed images (default: False)
        channel (int): Channel index for multi-channel images (default: None)
        calculate_circularity (bool): Calculate circularity metrics (default: False)
    
    Returns:
        tuple: Contains:
            - droplet_areas (list): Areas in pixels for each detected droplet
            - condensed_fraction (float): Percentage of image area covered by droplets
            - circularities (list, optional): Circularity values for each droplet
    """
    if (image_file.endswith('.tif') or image_file.endswith('.tiff')) and not image_file.startswith('.'):
        droplet_areas = []
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image: {image_path}")

        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        if original_image.dtype == np.uint16:
            original_image = (original_image / 256).astype(np.uint8)
        elif original_image.dtype != np.uint8:
            raise ValueError(f"Unsupported image type: {original_image.dtype}. Expected uint8 or uint16.")

        binarize_result = binarize(original_image, 
                                contrast=params['contrast'],  
                                block_size=params['block_size'],
                                display=display,
                                channel=channel,
                                calculate_circularity=calculate_circularity)
        
        if calculate_circularity:
            _, _, final_binary, circularities = binarize_result
        else:
            _, _, final_binary = binarize_result
            circularities = None
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(final_binary, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 0:
                droplet_areas.append(area)

        white_pixels = cv2.countNonZero(final_binary)
        condensed_fraction = 100 * (white_pixels) / (original_image.shape[0] * original_image.shape[1])
        print(f"White pixels: {white_pixels}, Percentage of image occupied by condensates: {condensed_fraction:.2f}%")

        if droplet_areas:
            avg_droplet_size = np.mean(droplet_areas)
            median_droplet_size = np.median(droplet_areas)
            num_droplets = len(droplet_areas)
                    
            print(f"Image: {image_file}")
            print(f"  Coverage: {condensed_fraction:.2f}%")
            print(f"  Number of droplets: {num_droplets}")
            print(f"  Average droplet area: {avg_droplet_size:.1f} pixels")
            print(f"  Median droplet area: {median_droplet_size:.1f} pixels")
            print(f"  Size range: {min(droplet_areas)} - {max(droplet_areas)} pixels")

            if calculate_circularity and circularities:
                avg_circularity = np.mean(circularities)
                median_circularity = np.median(circularities)
                print(f"  Average circularity: {avg_circularity:.3f}")
                print(f"  Median circularity: {median_circularity:.3f}")
                
    if calculate_circularity:
        return droplet_areas, condensed_fraction, circularities
    else:
        return droplet_areas, condensed_fraction


def process_folder(folder_path, params, display=False, channel=None, even_only=False, calculate_circularity=False):
    """
    Process all TIFF images in a folder using specified thresholding parameters.
    
    Args:
        folder_path (str): Path to folder containing TIFF images
        params (dict): Dictionary containing 'contrast' and 'block_size' parameters
        display (bool): Display processed images (default: False)
        channel (int): Channel index for multi-channel images (default: None)
        even_only (bool): Process only files with even-numbered suffixes (default: False)
        calculate_circularity (bool): Calculate circularity metrics (default: False)
    
    Returns:
        tuple: Contains:
            - folder_droplet_areas (list): List of droplet area lists (one per image)
            - folder_condensed_fractions (list): List of condensed fractions (one per image)
            - folder_circularities (list, optional): List of circularity lists (one per image)
    """
    folder_droplet_areas = []
    folder_condensed_fractions = []
    folder_circularities = [] if calculate_circularity else None

    def should_process_file(filename):
        if even_only:
            numbers = re.findall(r'\d+', filename)
            if numbers:
                return int(numbers[-1]) % 2 == 0
            return False
        return True
    
    tif_files = [f for f in os.listdir(folder_path) if (f.endswith('.tif') 
                or f.endswith('.tiff')) and not f.startswith('.') and should_process_file(f)]

    if not tif_files:
        subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
        for subfolder in subfolders:
            tif_files += [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.endswith(('.tif') or f.endswith('.tiff'))]

    for image_file in tif_files:
        if (image_file.endswith('.tif') or image_file.endswith('tiff')) and (not image_file.startswith('.')) and (not image_file.endswith("_RGB.tif")):
            result = process_file(image_file, folder_path, params, display, channel, calculate_circularity=calculate_circularity)
            if calculate_circularity:
                file_droplet_areas, file_condensed_fraction, file_circularities = result
                folder_circularities.append(file_circularities)
            else:
                file_droplet_areas, file_condensed_fraction = result
            folder_droplet_areas.append(file_droplet_areas)
            folder_condensed_fractions.append(file_condensed_fraction)
            
    if calculate_circularity:
        return folder_droplet_areas, folder_condensed_fractions, folder_circularities
    else:
        return folder_droplet_areas, folder_condensed_fractions


def process_single_folder(folder_path, contrast=-1, block_size=51, excludeSmallDots=0, 
                         output_csv_name=None, channel=None, display=False):
    """
    Process all images in a single folder and export results to CSV.
    
    This function is designed for analyzing images from a single experimental condition
    without requiring a config.json file.
    
    Args:
        folder_path (str): Path to folder containing TIFF images
        contrast (int): Constant subtracted from adaptive threshold mean (default: -1)
        block_size (int): Neighborhood size for threshold calculation, must be odd (default: 51)
        excludeSmallDots (int): Minimum area threshold for droplets in pixels (default: 0)
        output_csv_name (str): Name for output CSV file (default: auto-generated)
        channel (int): Channel index for multi-channel images (default: None)
        display (bool): Display processed images (default: False)
    
    Returns:
        pandas.DataFrame: DataFrame containing per-image metrics including:
            - image_name: Filename
            - condensed_fraction: Percentage of area covered
            - num_droplets: Number of detected droplets
            - avg_droplet_area: Mean droplet area
            - median_droplet_area: Median droplet area
            - total_droplet_area: Sum of all droplet areas
    """
    params = {
        'contrast': contrast,
        'block_size': block_size,
        'excludeSmallDots': excludeSmallDots
    }
    
    print(f"Processing single folder: {folder_path}")
    print(f"Parameters - Contrast: {contrast}, Block Size: {block_size}, Exclude Small Dots: {excludeSmallDots}")
    
    tif_files = [f for f in os.listdir(folder_path) 
                 if (f.endswith('.tif') or f.endswith('.tiff')) 
                 and not f.startswith('.') 
                 and not f.endswith("_RGB.tif")]
    
    if not tif_files:
        print("No TIFF files found in the specified folder.")
        return pd.DataFrame()
    
    results = []
    droplet_areas_all = []
    
    for image_file in tif_files:
        print(f"\nProcessing: {image_file}")
        
        droplet_areas, condensed_fraction = process_file(
            image_file, folder_path, params, display, channel
        )
        
        results.append({
            'image_name': image_file,
            'condensed_fraction': condensed_fraction,
            'num_droplets': len(droplet_areas),
            'avg_droplet_area': np.mean(droplet_areas) if droplet_areas else 0,
            'median_droplet_area': np.median(droplet_areas) if droplet_areas else 0,
            'total_droplet_area': sum(droplet_areas) if droplet_areas else 0
        })
        
        droplet_areas_all.extend(droplet_areas)
    
    df = pd.DataFrame(results)
    
    if output_csv_name is None:
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        output_csv_name = f"{folder_name}_condensed_fractions.csv"
    
    if not output_csv_name.endswith('.csv'):
        output_csv_name += '.csv'
    
    output_path = os.path.join(folder_path, output_csv_name)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*50}")
    print(f"SUMMARY FOR FOLDER: {os.path.basename(folder_path)}")
    print(f"{'='*50}")
    print(f"Images processed: {len(tif_files)}")
    print(f"Average condensed fraction: {df['condensed_fraction'].mean():.2f}% ± {df['condensed_fraction'].std():.2f}%")
    print(f"Range: {df['condensed_fraction'].min():.2f}% - {df['condensed_fraction'].max():.2f}%")
    print(f"Total droplets detected: {df['num_droplets'].sum()}")
    if droplet_areas_all:
        print(f"Overall avg droplet area: {np.mean(droplet_areas_all):.1f} pixels")
        print(f"Overall median droplet area: {np.median(droplet_areas_all):.1f} pixels")
    print(f"Data exported to: {output_path}")
    print(f"{'='*50}")
    
    return df


def process_all_concentrations(file_path, channel=None, even_only=False, calculate_circularity=False):
    """
    Process all subfolders specified in a config.json file.
    
    Each subfolder should contain images from a specific experimental condition and will be
    processed using the thresholding parameters specified in the config file.
    
    Args:
        file_path (str): Path to the directory containing subfolders and config.json
        channel (int): Channel index for multi-channel images (default: None)
        even_only (bool): Process only files with even-numbered suffixes (default: False)
        calculate_circularity (bool): Calculate circularity metrics (default: False)
    
    Returns:
        tuple: Contains:
            - droplet_areas_for_all_concentrations (list): Nested list of droplet areas
            - condensed_fractions_for_all_concentrations (list): Nested list of condensed fractions
            - circularities_for_all_concentrations (list, optional): Nested list of circularities
    """
    config_path = os.path.join(file_path, "config.json")

    droplet_areas_for_all_concentrations = []
    condensed_fractions_for_all_concentrations = []
    circularities_for_all_concentrations = [] if calculate_circularity else None

    with open(config_path, 'r') as file:
        config = json.load(file)

    for folder_name, params in config.items():
        print(f"Processing folder:", folder_name)
        print(f"Concentration: {folder_name}, Contrast: {params['contrast']}, Block Size: {params['block_size']}")

        folder_path = os.path.join(file_path, folder_name)
        result = process_folder(folder_path, params, channel, even_only=even_only, calculate_circularity=calculate_circularity)

        if calculate_circularity:
            droplet_areas, condensed_fractions, circularities = result
            circularities_for_all_concentrations.append(circularities)
        else:
            droplet_areas, condensed_fractions = result

        droplet_areas_for_all_concentrations.append(droplet_areas)
        condensed_fractions_for_all_concentrations.append(condensed_fractions)
        
    if calculate_circularity:
        return droplet_areas_for_all_concentrations, condensed_fractions_for_all_concentrations, circularities_for_all_concentrations   
    else:
        return droplet_areas_for_all_concentrations, condensed_fractions_for_all_concentrations


# ==============================================================================
# VISUALIZATION AND EXPORT FUNCTIONS
# ==============================================================================

def box_plot(droplet_data, fraction_data, folder_path, log_scale=False, plot_average_per_fov=False, circularity_data=None):
    """
    Generate side-by-side boxplots for droplet size and condensed fraction distributions.
    
    Creates a figure with 2-3 subplots showing:
    - Left: Droplet area distributions (or average per field of view)
    - Center: Condensed fraction distributions
    - Right (optional): Circularity distributions
    
    Args:
        droplet_data (list): Nested list of droplet areas for each condition
        fraction_data (list): Nested list of condensed fractions for each condition
        folder_path (str): Path to data folder (used to extract condition labels)
        log_scale (bool): Use logarithmic scale for droplet area plot (default: False)
        plot_average_per_fov (bool): Plot average droplet size per image instead of individual droplets (default: False)
        circularity_data (list): Nested list of circularity values (default: None)
    """
    concentrations = get_sorted_concentration(folder_path)
    positions = list(range(len(concentrations)))
    scatter_width = 0.25
    box_plot_width = 0.5

    num_plots = 3 if circularity_data is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
    
    if num_plots == 2:
        ax1, ax2 = axes
        
    all_folder_data = []

    for i, folder_areas in enumerate(droplet_data):
        concentration = concentrations[i]
        
        if plot_average_per_fov:
            fov_averages = []
            for image_areas in folder_areas:
                if image_areas:
                    fov_averages.append(np.mean(image_areas))
            
            all_folder_data.append(fov_averages)
            
            if fov_averages:
                scatter_positions = np.full(len(fov_averages), positions[i], dtype=float)
                scatter_positions += np.clip(np.random.normal(0, scatter_width, len(fov_averages)), 
                                           -scatter_width, scatter_width)
                ax1.scatter(scatter_positions, fov_averages, alpha=0.5, 
                           label=f"{concentration} µM", s=30)
        else:
            all_droplets = []
            for image_areas in folder_areas:
                all_droplets.extend(image_areas)

            all_folder_data.append(all_droplets)
            
            scatter_positions = np.full(len(all_droplets), positions[i], dtype=float)
            scatter_positions += np.clip(np.random.normal(0, scatter_width, len(all_droplets)), 
                                       -scatter_width, scatter_width)
            ax1.scatter(scatter_positions, all_droplets, alpha=0.5, 
                       label=f"{concentration} µM", s=30)

    ax1.boxplot(all_folder_data, positions=positions, widths=box_plot_width, showfliers=False)

    ax1.set_xlabel("Concentration (µM)")
    
    if log_scale:
        ax1.set_yscale('log')
        if plot_average_per_fov:
            ax1.set_ylabel("Average Droplet Area per FOV (log pixels)")
        else:
            ax1.set_ylabel("Droplet Area (log pixels)")
    else:
        if plot_average_per_fov:
            ax1.set_ylabel("Average Droplet Area per FOV (pixels)")
        else:
            ax1.set_ylabel("Droplet Area (pixels)")
    
    if plot_average_per_fov:
        ax1.set_title("Average droplet area per FOV by concentration")
    else:
        ax1.set_title("Droplet area distribution by concentration")
        
    ax1.set_xticks(positions)
    ax1.set_xticklabels([str(conc) for conc in concentrations])
    ax1.grid(True, alpha=0.3)
    
    if fraction_data and any(len(folder_fractions) > 0 for folder_fractions in fraction_data):
        non_empty_fractions = [folder_fractions for folder_fractions in fraction_data 
                              if len(folder_fractions) > 0]
        
        if non_empty_fractions:
            positions = list(range(1, len(non_empty_fractions) + 1))
            
            box_plots2 = ax2.boxplot(non_empty_fractions, positions=positions,
                                    patch_artist=True)
            
            colors2 = plt.cm.plasma(np.linspace(0, 1, len(non_empty_fractions)))
            for patch, color in zip(box_plots2['boxes'], colors2):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            
            ax2.set_xlabel('Concentration (µM)')
            ax2.set_ylabel('Condensed fraction (%)')
            ax2.set_title('Condensed fraction distribution by concentration')
            ax2.grid(True, alpha=0.3)
            
            ax2.set_xticks(positions)
            ax2.set_xticklabels([str(conc) for conc in concentrations])
    else:
        ax2.text(0.5, 0.5, 'No fraction data available', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Condensed fraction distribution by concentration')

    plt.tight_layout()
    plt.show()


def export_data_for_r(condensed_fractions, concentrations, file_path_and_name):
    """
    Export condensed fraction data to CSV format for statistical analysis.
    
    Creates a CSV with one row per image containing:
    - concentration_index: Index of the experimental condition
    - concentration: Numerical concentration value
    - image_number: Image index within the condition
    - condensed_fraction: Percentage of area covered by droplets
    
    Args:
        condensed_fractions (list): Nested list of condensed fractions by condition
        concentrations (list): List of concentration values corresponding to each condition
        file_path_and_name (str): Output CSV file path
    
    Returns:
        pandas.DataFrame: DataFrame containing the exported data
    """
    print(f"Number of concentrations: {len(concentrations)}")
    print(f"Concentrations: {concentrations}")
    print(f"Condensed fractions structure: {len(condensed_fractions)} replicates")
    
    for i, replicate in enumerate(condensed_fractions):
        print(f"Replicate {i}: {len(replicate)} data points")
    
    data_rows = []
    
    for conc_idx, images_for_concentration in enumerate(condensed_fractions):
        concentration_value = concentrations[conc_idx]
        
        for image_idx, condensed_fraction in enumerate(images_for_concentration):
            data_rows.append({
                'concentration_index': conc_idx,
                'concentration': concentration_value,
                'image_number': image_idx + 1,
                'condensed_fraction': condensed_fraction
            })
    
    df = pd.DataFrame(data_rows)
    df.to_csv(file_path_and_name, index=False)
    return df


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # Example usage
    folders_path = "path/to/your/data/folder"
    
    # Process all folders using config.json
    droplet_areas, condensed_fractions = process_all_concentrations(folders_path)
    box_plot(droplet_areas, condensed_fractions, folders_path, log_scale=False)
    
    # Export data for statistical analysis in R
    # concentrations = get_sorted_concentration(folders_path)
    # export_data_for_r(condensed_fractions, concentrations, "output_data.csv")
    
    # Preview binarization on a single image
    # test_image = cv2.imread("path/to/test/image.tif", cv2.IMREAD_GRAYSCALE)
    # binarize(test_image, contrast=-1, block_size=101, display=True)