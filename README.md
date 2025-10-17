# Condensate-Analysis

## Adaptive Threshold Condensate Analysis Pipeline

This repository provides tools for analyzing microscopy images of biomolecular condensates using adaptive thresholding. The pipeline segments droplets and quantifies properties such as area, condensed fraction, and circularity.

The analysis suite includes:
* GUI Application (condensate_analysis.py): Interactive parameter optimization and visualization
* Workflow Module (adaptive_threshold.py): Batch processing and statistical analysis

---

### Table of Contents
* Prerequisites and Setup
* GUI Application
* Workflow Module
* Data Structure
* Usage Examples

### Prerequisites and Setup

To run the analysis, ensure you have the necessary environment and data structure.

#### System Requirements

* Python 3.7+
* Recommended: virtual environment for package management.

#### Required Libraries: 

Install the dependencies using pip:

```
pip install numpy pandas opencv-python matplotlib pillow scipy
```

For GUI functionality, ensure tkinter is available (typically included with python).
```
# Test tkinter installation
python -m tkinter
```
---
### GUI Application

The GUI (condensate_analysis.py) provides an interactive interface for:
* Loading single or multichannel microscopy images.
* Real time parameter adjustment (contrast, block size, brightness).
* Visual comparison of original, contour, and binary images.
* Region of interest (ROI) analysis.
* Parameter sweep PDF generation for batch validation.

#### Launching the GUI
```
python condensate_analysis.py
```

### Data Structure

Microscopy images must be organized into subfolders, with each folder representing a single experimental condition (e.g., a specific concentration).

    /data_root_folder
    
        ├── /5.0uM
    
            ├── image_01.tif
    
            └── image_02.tif
    
        ├── /10.0uM
    
            ├── image_01.tif
    
            └── image_02.tif
    
        └── config.json

### Configuration File

The config.json file, placed in the root data directory, specifies the contrast and block_size parameters for adaptive thresholding for each condition folder. These values should be determined first using the GUI tool (see next section).

Example config.json:

```
{
  "5.0uM": {
    "contrast": -1,
    "block_size": 51
  },
  "10.0uM": {
    "contrast": -3,
    "block_size": 41
  }
}
```

## Usage
The script's primary function is to process all conditions defined in the config.json.

### Running the Full Analysis
Set the path to your data folder in the if __name__ == "__main__": block to begin processing:
```
# Example from condensate_workflow.py
folders_path = "path/to/your/data/folder"
droplet_areas, condensed_fractions = process_all_concentrations(folders_path)
box_plot(droplet_areas, condensed_fractions, folders_path, log_scale=False)
```
### Single Folder Mode
The process_single_folder function is available for processing a single directory without requiring a config.json file. It exports a summary CSV directly to the folder.
```
# Example: Process one folder and export a CSV immediately
process_single_folder(
    folder_path="path/to/single/condition/folder",
    contrast=-1,
    block_size=51,
    output_csv_name="single_condition_results.csv"
 )
```
### Data Export for R
The script includes visualization functions, but for statistical analysis, it's recommended to export the raw data.

Use the export_data_for_r function to generate a tidy CSV file suitable for import into R or other statistical environments. This format provides one row per image, detailing the concentration and calculated condensed fraction for use in standard statistical tests and visualisation.
```
# Example of exporting data in the __main__ block
concentrations = get_sorted_concentration(folders_path)
_, condensed_fractions = process_all_concentrations(folders_path)
export_data_for_r(condensed_fractions, concentrations, "output_data_for_r.csv")
```
The resulting CSV will contain columns such as:
* concentration
* image_number
* condensed_fraction
