import os
import json
import logging
import argparse
import random
import string
import zipfile
import requests

import torch
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt

from glob import glob
from typing import List, Tuple, Union, Optional
from skimage.measure import label
from scipy.ndimage import binary_dilation
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from batchgenerators.utilities.file_and_folder_operations import join
from nibabel.orientations import io_orientation, inv_ornt_aff, apply_orientation

from functools import partial
import signal
import multiprocessing
import sys

# Global variable to track the multiprocessing pool 
pool = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def signal_handler(signum, frame): #--> avoid zombie processes
    """
    Handle termination signals like SIGTERM and SIGINT (Ctrl+C).
    """
    global pool
    print(f"\nReceived signal {signum}. Cleaning up...")
    if pool:
        pool.terminate()
        pool.join()
    sys.exit(1)

def configure_environment_variables():
    """
    Set environment variables required for the nnU-Net models.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nnunet_models_dir = os.path.join(base_dir, '../models')
    nnunet_cmd_dir = os.path.join(base_dir, '../cmd')
    os.environ['NNUNET_MODELS_DIR'] = nnunet_models_dir
    os.environ['NNUNET_CMD_DIR'] = nnunet_cmd_dir

    logging.info(f"NNUNET_MODELS_DIR is set to {nnunet_models_dir}")
    logging.info(f"NNUNET_CMD_DIR is set to {nnunet_cmd_dir}")


def get_device() -> str:
    """Determine and return the available device for computation."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def read_image(file_path):
    """
    Read a NIfTI image, optionally reorient to RAS using SimpleITK, and return the image and properties.
    
    Parameters:
    file_path (str): Path to the NIfTI image file.
    reorient (bool): Whether to reorient the image to RAS. Default is True.
    
    Returns:
    SimpleITK.Image: The reoriented or original image as a SimpleITK image.
    dict: Properties including original and reoriented directions and origins.
    """
    try:
        # Read the image using SimpleITK
        sitk_image = sitk.ReadImage(file_path)
        
        # Store original direction and origin
        original_direction = sitk_image.GetDirection()
        original_origin = sitk_image.GetOrigin()
        #reoriented_image = sitk.DICOMOrient(sitk_image, "RAS")
        reoriented_image = sitk_image
        reoriented_direction = reoriented_image.GetDirection()
        reoriented_origin = reoriented_image.GetOrigin()
        
        logging.debug(f'Original Direction: {original_direction} Reoriented direction {reoriented_direction}')
        
    except Exception as e:
        print(f"Error reading image: {e}")
        raise RuntimeError("Failed to read or reorient the image.")

    return reoriented_image


def write_image(sitk_image, file_path):
    """
    Save a SimpleITK image, optionally adjusting direction and origin.
    
    Parameters:
    sitk_image (SimpleITK.Image): The SimpleITK image to be saved.
    file_path (str): Path to save the NIfTI image file.
    properties (dict): Properties including original direction and origin.
    """
    try:
        # Save the image
        sitk.WriteImage(sitk_image, file_path)
        print(f"Image saved to {file_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
        raise RuntimeError("Failed to save the image.")


def extract_spacing_from_plans(file_path: str) -> Union[List[float], str]:
    """
    Extract and return the spacing from a plan configuration file.
    
    Parameters:
        file_path (str): Path to the configuration file.
        
    Returns:
        list or str: Spacing values in reversed order or an error message.
    """
    config_key = os.path.basename(os.path.dirname(file_path)).split("__")[-1]
    logging.debug(f"Extracted config key: {config_key}")

    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            spacing = data['configurations'][config_key]['spacing']
            logging.debug(f"Spacing found: {spacing}")
            return spacing[::-1]
    except KeyError:
        error_msg = f"Configuration '{config_key}' not found in the file."
        logging.error(error_msg)
        return error_msg


def fetch_label_value_from_json(model_path: str, labelname: str) -> int:
    """
    Retrieve the label value for a given label name from a JSON file.
    
    Parameters:
        model_path (str): Path to the model directory.
        labelname (str): Label name to lookup.
        
    Returns:
        int: Label value or -1 if not found.
    """
    file_path = os.path.join(model_path, 'dataset.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['labels'].get(labelname, -1)


def fetch_orientation_from_json(model_path: str) -> bool:
    """
    Retrieve the overwrite_image_reader_writer value from a dataset JSON file.
    
    Parameters:
        model_path (str): Path to the model directory.
        
    Returns:
        bool: True if "reorient" is in overwrite_image_reader_writer, False otherwise.
    """
    file_path = os.path.join(model_path, 'dataset.json')
    with open(file_path, 'r') as file:
        data = json.load(file)

    overwrite_value = data.get('overwrite_image_reader_writer', "")
    
    return "reorient" in overwrite_value.lower()


def fetch_label_name_from_json(model_path: str, value: int) -> str:
    """
    Retrieve the label name for a given value from a JSON file.
    
    Parameters:
        model_path (str): Path to the model directory.
        value (int): Value to lookup.
        
    Returns:
        str: Label name or 'Unknown label' if not found.
    """
    file_path = os.path.join(model_path, 'dataset.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    reverse_lookup = {v: k for k, v in data['labels'].items()}
    return reverse_lookup.get(value, "Unknown label")


def load_models_from_json(json_file: str) -> Tuple[List[dict], pd.DataFrame]:
    """
    Load model configurations from a JSON file.
    
    Parameters:
        json_file (str): Path to the JSON file.
        
    Returns:
        tuple: List of models and DataFrame with model information.
    """
    if os.getenv('NNUNET_MODELS_DIR') is None:
        raise ValueError("The environment variable 'NNUNET_MODELS_DIR' is not set.")
    if os.getenv('NNUNET_CMD_DIR') is None:
        raise ValueError("The environment variable 'NNUNET_CMD_DIR' is not set.")
    
    json_path = os.path.join(os.getenv('NNUNET_CMD_DIR'), f"{json_file}.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    models = []
    records = []
    for dataset, details in data.items():
        model_path = os.path.join(os.getenv('NNUNET_MODELS_DIR'), dataset)
        order = details['order']
        trainer = details['trainer']
        fold = details['fold']
        checkpoint = details['checkpoint']
        config = details['config']
        labels = details['labels']
        datasetjsonpath = os.path.join(model_path, "__".join([trainer, 'nnUNetPlans', config]))

        # Append the model entry only once per dataset
        model = {
            'path': model_path,
            'trainer': trainer,
            'config': config,
            'fold': fold,
            'checkpoint_name': checkpoint + '.pth',
        }
        models.append(model)
        
        # Add entries to the records for each label
        for label_name, label_value in labels.items():
            record = {
                'ORDER': order,
                'DATASET': dataset,
                'TRAINER': trainer,
                'FOLD': fold,
                'CHECKPOINT': checkpoint,
                'CONFIG': config,
                'LABELNAME': label_name,
                'LABEL': fetch_label_value_from_json(datasetjsonpath, label_name),  # corrected
                'MULTILABEL': label_value
            }
            records.append(record)

    df = pd.DataFrame(records)
    return models, df


def load_images(images: List[sitk.Image]) -> Tuple[np.ndarray, dict]:
    """
    Read medical images and extract metadata.
    
    Parameters:
        images (list): List of ITK images to process.
        
    Returns:
        tuple: Stacked numpy images and a dictionary with image metadata.
    """
    np_images, spacings, origins, directions, spacings_for_nnunet = [], [], [], [], []

    for img in images:
        spacings.append(img.GetSpacing())
        origins.append(img.GetOrigin())
        directions.append(img.GetDirection())

        npy_image = sitk.GetArrayFromImage(img)[None, ...]  # Adding a new axis at position 0
        np_images.append(npy_image)

        nnunet_spacing = list(spacings[-1])[::-1]  # Reverse the order of spacings
        spacings_for_nnunet.append([abs(s) for s in nnunet_spacing])

        logging.debug(f"Processed image with spacing: {spacings[-1]}")

    stacked_images = np.vstack(np_images)
    metadata = {
        'sitk_stuff': {
            'spacing': spacings[0],
            'origin': origins[0],
            'direction': directions[0]
        },
        'spacing': spacings_for_nnunet[0]
    }

    return stacked_images.astype(np.float32), metadata


def rescale_itk_image(itk_image: sitk.Image, target_spacing: Tuple[float, float, float]) -> sitk.Image:
    """
    Rescale the input SimpleITK image to the specified target spacing.
    
    Parameters:
        itk_image (SimpleITK.Image): The input SimpleITK image.
        target_spacing (tuple): Target spacing as a tuple (x, y, z).
        
    Returns:
        SimpleITK.Image: Rescaled SimpleITK image.
    """
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    new_size = [int(round(osz * osp / tsz)) for osz, osp, tsz in zip(original_size, original_spacing, target_spacing)]

    logging.debug(f"Original size: {original_size}, Original spacing: {original_spacing}")
    logging.debug(f"Target spacing: {target_spacing}, New size: {new_size}")

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear) #sitkBSpline
    resampler.SetOutputOrigin(itk_image.GetOrigin())
    resampler.SetOutputDirection(itk_image.GetDirection())

    rescaled_image = resampler.Execute(itk_image)
    return rescaled_image


def split_itk_image(image: sitk.Image, margin: int = 20) -> List[sitk.Image]:
    """
    Split the input image into three overlapping parts along the z-axis using SimpleITK.
    
    Parameters:
        image (SimpleITK.Image): Input ITK image.
        margin (int): Margin for overlapping. Default is 20.
        
    Returns:
        list: List of ITK images representing the subparts.
    """
    img_array = sitk.GetArrayFromImage(image)
    z_dim = img_array.shape[0]
    third = z_dim // 3

    parts = [
        sitk.GetImageFromArray(img_array[:third + margin, :, :]),
        sitk.GetImageFromArray(img_array[third - margin:2 * third + margin, :, :]),
        sitk.GetImageFromArray(img_array[2 * third - margin:, :, :])
    ]

    for part in parts:
        part.SetSpacing(image.GetSpacing())
        part.SetOrigin(image.GetOrigin())
        part.SetDirection(image.GetDirection())

    return parts


def stitch_itk_image(parts: List[sitk.Image], margin: int = 20) -> sitk.Image:
    """
    Stitch the input image parts back together along the z-axis using SimpleITK,
    removing the overlapping regions.
    
    Parameters:
        parts (list): List of ITK images representing the subparts.
        margin (int): Margin used for overlapping. Default is 20.
        
    Returns:
        SimpleITK.Image: Stitched ITK image.
    """
    part_arrays = [sitk.GetArrayFromImage(part) for part in parts]

    if part_arrays[1].shape[0] - 2 * margin < 0:
        raise ValueError("Margin is too large for image size. Reduce the margin size.")

    length_1 = part_arrays[0].shape[0] - margin
    length_2 = part_arrays[1].shape[0] - 2 * margin
    length_3 = part_arrays[2].shape[0] - margin

    non_overlap_arrays = [
        part_arrays[0][:length_1, :, :],
        part_arrays[1][margin:margin + length_2, :, :],
        part_arrays[2][margin:, :, :]
    ]

    stitched_array = np.concatenate(non_overlap_arrays, axis=0)

    stitched_image = sitk.GetImageFromArray(stitched_array)
    stitched_image.SetSpacing(parts[0].GetSpacing())
    stitched_image.SetOrigin(parts[0].GetOrigin())
    stitched_image.SetDirection(parts[0].GetDirection())

    return stitched_image


def extract_largest_connected_component(input_image: sitk.Image) -> sitk.Image:
    """
    Extract and return the largest connected component from the input SimpleITK image.
    If the input image is empty, return the input image.
    
    Parameters:
        input_image (SimpleITK.Image): The input image.
        
    Returns:
        SimpleITK.Image: Image containing only the largest connected component, or the input image if it's empty.
    """
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    connected_components = connected_component_filter.Execute(input_image)

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(connected_components)

    labels = label_shape_filter.GetLabels()
    if not labels:
        logging.warning("Input image is empty (contains no foreground pixels). Returning the input image.")
        return input_image

    largest_label = max(labels, key=label_shape_filter.GetNumberOfPixels)
    logging.debug(f"Largest label identified: {largest_label} out of {len(labels)} with {label_shape_filter.GetNumberOfPixels(largest_label)} pixels.")

    largest_component = sitk.BinaryThreshold(connected_components, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    return largest_component


def remove_small_components(input_image: sitk.Image, min_size: int) -> sitk.Image:
    """
    Remove all components smaller than the given threshold from the input image.
    
    Parameters:
        input_image (SimpleITK.Image): The input image.
        min_size (int): The minimum size (in voxels) for a component to be kept.
        
    Returns:
        SimpleITK.Image: The image with small components removed.
    """
    connected_component_filter = sitk.ConnectedComponentImageFilter()
    connected_components = connected_component_filter.Execute(input_image)
    
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(connected_components)

    labels = label_shape_filter.GetLabels()
    output_image = sitk.Image(input_image.GetSize(), sitk.sitkUInt8)
    output_image.CopyInformation(input_image)

    for label in labels:
        if label_shape_filter.GetNumberOfPixels(label) >= min_size:
            component_mask = sitk.BinaryThreshold(connected_components, lowerThreshold=label, upperThreshold=label, insideValue=1, outsideValue=0)
            output_image = output_image | component_mask
    
    return output_image

def remove_small_components_multilabel(input_image: sitk.Image, min_size: int) -> sitk.Image:
    """
    Remove all components smaller than the given threshold from the input multi-label image,
    and also remove small holes by processing the inverse of the label.
    
    Parameters:
        input_image (SimpleITK.Image): The input multi-label image.
        min_size (int): The minimum size (in voxels) for a component to be kept.
        
    Returns:
        SimpleITK.Image: The image with small components and small holes removed.
    """
    # Get unique labels in the image
    unique_labels = np.unique(sitk.GetArrayViewFromImage(input_image))

    # Create an empty output image
    output_image = sitk.Image(input_image.GetSize(), input_image.GetPixelID())
    output_image.CopyInformation(input_image)

    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        # Create a binary image for the current label
        binary_image = sitk.BinaryThreshold(input_image, lowerThreshold=float(label), upperThreshold=float(label), insideValue=int(1), outsideValue=int(0))
        
        # Remove small components in the binary image
        filtered_binary_image = remove_small_components(binary_image, min_size)
        
        # Process the inverse to remove small holes
        inverse_binary_image = sitk.InvertIntensity(filtered_binary_image, maximum=1)
        filtered_inverse_binary_image = remove_small_components(inverse_binary_image, min_size)
        final_binary_image = sitk.InvertIntensity(filtered_inverse_binary_image, maximum=1)
        
        # Convert the final binary image back to the original label
        filtered_label_image = sitk.BinaryThreshold(final_binary_image, lowerThreshold=float(1), upperThreshold=float(1), insideValue=int(label), outsideValue=int(0))
        
        # Cast to same type
        filtered_label_image = sitk.Cast(filtered_label_image, output_image.GetPixelID())

        # Combine the filtered label image with the output image
        output_image = sitk.Add(output_image, filtered_label_image)

    return output_image

def repair_multilabel(segmentation_image: sitk.Image, min_size: int) -> sitk.Image:
    """
    Repair the remaining components in the segmentation image by first removing small components and then
    assigning larger remaining components the label of their best-connected neighbor.

    Parameters:
        segmentation_image (SimpleITK.Image): The input segmentation image (multi-label).
        min_size (int): The minimum size (in voxels) for a component to be kept.
        
    Returns:
        sitk.Image: The repaired segmentation image.
    """
    segmentation_array = sitk.GetArrayFromImage(segmentation_image)
    repaired_array = np.zeros_like(segmentation_array,dtype=int)
    switching_label_tracker = np.zeros_like(segmentation_array,dtype=int)

    unique_labels = np.unique(segmentation_array[segmentation_array>0])

    for label in unique_labels:
        label_mask = segmentation_array == label
        # Remove small components and set largest CC as label
        label_mask_image = sitk.GetImageFromArray(label_mask.astype(np.uint8))
        label_mask_image.CopyInformation(segmentation_image)

        largest_label_mask_image = extract_largest_connected_component(label_mask_image)
        largest_label_mask_array = sitk.GetArrayFromImage(largest_label_mask_image).astype(bool)
        repaired_array[largest_label_mask_array] = label

        remaining_components_mask = (label_mask & ~largest_label_mask_array)
        remaining_components_image = sitk.GetImageFromArray(remaining_components_mask.astype(int))

        cleaned_label_mask_image = remove_small_components(remaining_components_image, min_size)
        cleaned_label_mask_array = sitk.GetArrayFromImage(cleaned_label_mask_image).astype(bool)
        
        if np.any(cleaned_label_mask_array):
            # Find connected components in the cleaned mask
            cleaned_label_mask_image = sitk.GetImageFromArray(cleaned_label_mask_array.astype(np.uint8))
            cleaned_label_mask_image.CopyInformation(segmentation_image)
            connected_component_filter = sitk.ConnectedComponentImageFilter()
            connected_components = connected_component_filter.Execute(cleaned_label_mask_image)
            
            label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
            label_shape_filter.Execute(connected_components)
            remaining_labels = label_shape_filter.GetLabels()
            
            for remaining_label in remaining_labels:
                component_mask = sitk.BinaryThreshold(connected_components, lowerThreshold=remaining_label, upperThreshold=remaining_label, insideValue=1, outsideValue=0)
                component_mask.CopyInformation(segmentation_image)
                component_array = sitk.GetArrayFromImage(component_mask).astype(bool)
                 
                # Find the best connected neighbor for this component in the original image
                dilated_mask = binary_dilation(component_array, iterations=1) & (component_array == 0)
                neighbor_labels = segmentation_array[dilated_mask & (segmentation_array > 0)]
                unique_neighbors, counts = np.unique(neighbor_labels, return_counts=True)

                if len(unique_neighbors) > 0:
                    best_neighbor_label = unique_neighbors[np.argmax(counts)]
                    logging.debug(f'Label {label}: Repairing component #{remaining_label}, best neighbour: {best_neighbor_label}')
                    repaired_array[component_array] = best_neighbor_label
                    
                    # This tracks which labels have been re-assigned. If a label is re-assigned multiple times a warning is raised
                    # This typically means that there is some problem in the original labels. E.g. Missing, Split or Duplicate labels
                    # Update the switching label tracker
                    switching_label_tracker[component_array] += 1
                    if np.any(switching_label_tracker[component_array] > 1):
                        logging.warning(f'There seems to be a duplicate label/unconnected component for label: {label} that cannot be resolved')

                else:
                    logging.debug(f'Label {label}: has no neigbours')
                    repaired_array[component_array] = 0
    
    repaired_image = sitk.GetImageFromArray(repaired_array)
    repaired_image.CopyInformation(segmentation_image)
    
    return repaired_image


def extract_largest_connected_component_multilabel(input_image: sitk.Image) -> sitk.Image:
    """
    Extract and return the largest connected component for each label in the input SimpleITK image.
    If the input image is empty, return the input image.
    
    Parameters:
        input_image (SimpleITK.Image): The input image.
        repair (bool): If True, assigns removed components to the label of their best-connected neighbor. If none, assigns 0.
        
    Returns:
        SimpleITK.Image: Image containing only the largest connected component for each label.
    """
    # Create the output image
    logging.info(f'Running largest connected component (and repair) analysis...')

    output_image = sitk.Image(input_image.GetSize(), input_image.GetPixelID())
    output_image.CopyInformation(input_image)

    # Convert input image to numpy array for easier label-wise processing
    input_array = sitk.GetArrayFromImage(input_image)
    output_array = np.zeros_like(input_array)

    # Process each label separately
    unique_labels = np.unique(input_array)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]  # Exclude background label

    for label in unique_labels:
        logging.debug(f'Extracting single connected component for label {label}')
        # Extract the current label's binary mask
        label_mask = input_array == label
        sitk_label_mask = sitk.GetImageFromArray(label_mask.astype(np.uint8))
        sitk_label_mask.CopyInformation(input_image)
        
        # Apply largest connected component extraction
        largest_component_mask = extract_largest_connected_component(sitk_label_mask)
        largest_component_array = sitk.GetArrayFromImage(largest_component_mask).astype(bool)
        
        output_array[largest_component_array] = label
    
    output_image = sitk.GetImageFromArray(output_array)
    output_image.CopyInformation(input_image)
    
    return output_image


def perform_prediction(images: List[sitk.Image], path: str, trainer: str, config: str, checkpoint_name: str = 'checkpoint_final.pth', fold = 0, target_spacing: Optional[Tuple[float, float, float]] = None, force_split: bool = True) -> sitk.Image:
    """
    Predict the output for a given image, potentially splitting it into subparts if needed.
    
    Parameters:
        images (list): List of ITK images to process.
        path (str): Path to the model directory.
        trainer (str): Trainer name.
        config (str): Configuration name.
        checkpoint_name (str): Checkpoint file name. Default is 'checkpoint_final.pth'.
        fold (int): Fold number. Default is 0.
        target_spacing (tuple, optional): Target spacing. Default is None.
        force_split (bool): Flag to force splitting. Default is True.
        
    Returns:
        SimpleITK.Image: Predicted image.
    """
    model_path = os.path.join(path, "__".join([trainer, 'nnUNetPlans', config]))

    if target_spacing is None:
        target_spacing = extract_spacing_from_plans(os.path.join(model_path, 'plans.json'))
        logging.debug(f"Target spacing set to: {target_spacing}")

    rescaled_images = [rescale_itk_image(img, target_spacing) for img in images]

    img_arrays = [sitk.GetArrayFromImage(im) for im in rescaled_images]
    ss = img_arrays[0].shape
    nr_voxels_thr = 900 * 900 * 900 #512
    do_triple_split = np.prod(ss) > nr_voxels_thr and ss[0] > 200
    if force_split:
        do_triple_split = True

    if do_triple_split:
        logging.info("Splitting into subparts...")
        split_parts = [split_itk_image(im) for im in rescaled_images]

        predictions = []
        for i, parts in enumerate(zip(*split_parts)):
            logging.info(f"Processing part {i}")
            prediction_part = predict_with_nnunet(list(parts), model_path, checkpoint_name, fold)
            predictions.append(prediction_part)
        combined_prediction = stitch_itk_image(predictions)
    else:
        combined_prediction = predict_with_nnunet(rescaled_images, model_path, checkpoint_name, fold)

    combined_prediction.SetSpacing(rescaled_images[0].GetSpacing())
    combined_prediction.SetOrigin(rescaled_images[0].GetOrigin())
    combined_prediction.SetDirection(rescaled_images[0].GetDirection())

    logging.debug("Prediction complete.")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(images[0])
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    logging.debug("Resampling prediction to original image dimensions.")
    return resampler.Execute(combined_prediction)


def save_prediction_slice(image: sitk.Image):
    """
    Save the middle slice (along the z-axis) of the given 3D image as a PNG file
    with a random filename in the current working directory.
    
    Parameters:
        image (SimpleITK.Image): The 3D image to save a slice from.
    """
    # Convert the SimpleITK image to a numpy array
    np_image = sitk.GetArrayFromImage(image)
    
    # Get the middle slice index along the z-axis
    middle_index = np_image.shape[0] // 2
    
    # Extract the middle slice
    middle_slice = np_image[middle_index, :, :]
    
    # Generate a random filename
    random_filename = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) + '.png'
    
    # Save the middle slice as a PNG file in the current working directory
    plt.imsave(random_filename, middle_slice, cmap='gray')
    
    logging.info(f"Saved prediction slice as {random_filename}")


def predict_with_nnunet(images: List[sitk.Image], model_path: str, checkpoint_name: str = 'checkpoint_final.pth', fold=0) -> sitk.Image:
    """
    Predict output using a nnU-Net model for given images.
    
    Parameters:
        images (list): List of ITK images for prediction.
        model_path (str): Path to the trained nnUNet model directory.
        checkpoint_name (str): Checkpoint file name. Default is 'checkpoint_final.pth'.
        fold (int or list): Fold number(s). Default is 0.
        
    Returns:
        SimpleITK.Image: ITK image of the prediction.
    """
    logger = logging.getLogger()
    verbose = logger.getEffectiveLevel() == logging.DEBUG

    # Ensure fold is a list
    folds = [fold] if not isinstance(fold, list) else fold

    predictor = nnUNetPredictor(
        tile_step_size=0.8,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=torch.device(get_device()),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(model_path, use_folds=folds, checkpoint_name=checkpoint_name)

    img, props = load_images(images)

    prediction = predictor.predict_single_npy_array(img, props, None, None, False)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.SetSpacing(images[0].GetSpacing())
    prediction_image.SetOrigin(images[0].GetOrigin())
    prediction_image.SetDirection(images[0].GetDirection())

    return prediction_image

def crop_image_to_trunk(image: sitk.Image) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Crop the input image to the trunk region using a pre-trained nnU-Net model.
    
    Parameters:
        image (SimpleITK.Image): Input image to be cropped.
        
    Returns:
        tuple: ROI size and start index for cropping.
    """
    home = os.getenv('NNUNET_MODELS_DIR')
    model_dir = 'Dataset300*/nnUNetTrainer__nnUNetPlans__3d_fullres'
    model_paths = glob(os.path.join(home, model_dir))

    if len(model_paths)==0:
        logging.info(f'Model directory does not exist. Downloading the model...')
        url = "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip"
        download_model(home, url)  # You need to implement this function to handle the download
        model_paths = glob(os.path.join(home, model_dir))

    logging.info(f'Cropping to trunk using {model_paths[0]}')
    trunc = predict_with_nnunet([image], model_paths[0], checkpoint_name='checkpoint_final.pth', fold=0)

    binary_seg = sitk.Cast(trunc == 1, sitk.sitkUInt8)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(binary_seg)
    bounding_box = label_stats.GetBoundingBox(1)
    roi_size = bounding_box[3:6]
    roi_start = bounding_box[0:3]
    logging.debug(f"Original size: {image.GetSize()}")
    logging.debug(f"Cropped to size: {roi_size}, start: {roi_start}")
    return roi_size, roi_start

def crop_image_to_label(image: sitk.Image, seglabel: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Crop the input image to the trunk region using a pre-trained nnU-Net model.
    
    Parameters:
        image (SimpleITK.Image): Input image to be cropped.
        
    Returns:
        tuple: ROI size and start index for cropping.
    """

    binary_seg = sitk.Cast(image == seglabel, sitk.sitkUInt8)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(binary_seg)
    bounding_box = label_stats.GetBoundingBox(1)
    roi_size = bounding_box[3:6]
    roi_start = bounding_box[0:3]
    logging.debug(f"Original size: {image.GetSize()}")
    logging.debug(f"Cropped to size: {roi_size}, start: {roi_start}")
    return roi_size, roi_start

def download_model(model_path: str, url: str):
    """
    Download a zip file from a URL, extract its contents into the specified path,
    and delete the zip file after extraction.

    Parameters:
        model_path (str): The path where the model should be extracted.
    """
    # URL of the zip file to download
    
    # Create the folder if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    
    # Filename from the URL
    filename = os.path.join(model_path, url.split("/")[-1])
    
    # Download the zip file
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download from {url}. Status code: {response.status_code}")
        return
    
    # Unzip the file
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(model_path)
        print(f"Extracted contents to {model_path}")
    
    # Delete the zip file after extraction
    os.remove(filename)
    print(f"Deleted {filename}")
    

def download_models_from_file():
    # Get the directory path where the file is located
    configure_environment_variables()
    models_dir = os.getenv('NNUNET_MODELS_DIR')
    if not models_dir:
        print("Error: NNUNET_MODELS_DIR environment variable is not set.")
        return
    
    # Construct the file path
    file_path = os.path.join(models_dir, 'modelcollection.txt')
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    # Open and read each line in the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        # Process each line (assuming each line contains a URL)
        for line in lines:
            line = line.strip()  # Remove leading/trailing whitespace
            if line and not line.startswith('#'):  # Skip comments and empty lines
                parts = line.split(',', 1)
                if len(parts) == 2:
                    dataset_id, url = parts
                    download_model(models_dir, url.strip())


def fill_cropped_image_into_original(cropped_image: sitk.Image, original_image: sitk.Image, roi_size: Tuple[int, int, int], roi_start: Tuple[int, int, int]) -> sitk.Image:
    """
    Fill the cropped image into the original image dimensions based on ROI.
    
    Parameters:
        cropped_image (SimpleITK.Image): Cropped image.
        original_image (SimpleITK.Image): Original image.
        roi_size (tuple): ROI size.
        roi_start (tuple): ROI start index.
        
    Returns:
        SimpleITK.Image: Filled image.
    """
    pixel_type = sitk.sitkInt16
    cropped_image = sitk.Cast(cropped_image, pixel_type)
    original_image = sitk.Cast(original_image, pixel_type)

    # Verify ROI is within original image bounds
    original_size = original_image.GetSize()
    if any(roi_start[i] + roi_size[i] > original_size[i] for i in range(3)):
        raise ValueError("ROI extends beyond the original image dimensions.")

    empty_img = sitk.Image(original_image.GetSize(), pixel_type)
    empty_img.CopyInformation(original_image)

    paste_filter = sitk.PasteImageFilter()
    paste_filter.SetDestinationIndex(roi_start)
    paste_filter.SetSourceSize(roi_size)  # Ensure source size matches ROI size

    filled_image = paste_filter.Execute(empty_img, cropped_image)

    # Log where the cropped image is pasted
    logging.debug(f"Filled cropped image into original at ROI start: {roi_start} with size: {roi_size}.")
    logging.debug(f"Destination index set to: {roi_start}")
    logging.debug(f"Source size set to: {roi_size}")
    logging.debug(f"Restoring size from: {cropped_image.GetSize()} to: {original_image.GetSize()}")

    return filled_image


def process_image_with_model(images: List[sitk.Image], model: dict, df: pd.DataFrame, force_split: bool = True) -> List[Tuple[sitk.Image, int]]:
    """
    Process an image using the specified model.
    
    Parameters:
        images (list): List of ITK images to process.
        model (dict): Model configuration.
        df (DataFrame): DataFrame with model information.
        force_split (bool): Flag to force splitting. Default is True.
        
    Returns:
        list: Processed segments.
    """
    segmentation = perform_prediction(
        images, 
        model['path'], 
        model['trainer'], 
        model['config'], 
        checkpoint_name=model['checkpoint_name'], 
        fold=model['fold'], 
        target_spacing=None, 
        force_split=force_split
    )
    processed_segments = []
    for label in np.unique(sitk.GetArrayFromImage(segmentation)):
        if label > 0:
            label_row = df[(df['DATASET'] == os.path.basename(model['path'])) & (df['LABEL'] == label)]
            if label_row.empty:
                logging.info(f'Labelrow empty: {label}')
                continue  # Skip if the label does not exist in the DataFrame
            binary_segmentation = segmentation == label
            # Assign new multilabel labels
            multilabel_value = label_row['MULTILABEL'].values[0]
            binary_segmentation = sitk.Cast(binary_segmentation, sitk.sitkUInt16) * multilabel_value
            processed_segments.append((binary_segmentation, label))
    return processed_segments


def save_segmentation_result(output_path: str, image_base: str, model_base: str, segmentation: sitk.Image, label_names: dict, multilabel: bool, name: Optional[str] = None):
    """
    Save the segmentation to the specified output path.
    
    Parameters:
        output_path (str): Path to the output directory.
        image_base (str): Base name of the image.
        model_base (str): Base name of the model.
        segmentation (SimpleITK.Image): Segmentation image.
        label_names (dict): Dictionary of label names.
        multilabel (bool): Flag to enable multilabel mode.
        name (str, optional): Optional name for saving the segmentation.
    """

    if multilabel:
        logging.info('Saving multilabel file')
        filename = f"{name}.nii.gz"
        output_name = os.path.join(output_path, image_base, f'{image_base}_{filename}')
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        logging.info(f'Saving {output_name}')
        segmentation = sitk.Cast(segmentation, sitk.sitkUInt16)
        write_image(segmentation, output_name)
    else:
        segmentation_array = sitk.GetArrayFromImage(segmentation)
        labels = np.unique(segmentation_array[segmentation_array > 0])
        for label in labels:
            if label in label_names:
                filename = f"{label_names[label]}.nii.gz"
                output_name = os.path.join(output_path, image_base, f'{image_base}_{model_base}_{filename}')
                os.makedirs(os.path.dirname(output_name), exist_ok=True)
                logging.info(f'Saving {output_name}')
                label_image_array = (segmentation_array == label).astype(np.uint8)
                if np.any(label_image_array):
                    label_image = sitk.GetImageFromArray(label_image_array)
                    label_image.CopyInformation(segmentation)
                    label_image = sitk.Cast(label_image, sitk.sitkUInt16)
                    write_image(label_image, output_name)
                    logging.debug(f'Successfully saved {output_name} with label {label}')
                else:
                    logging.warning(f"Label {label} has no data and will not be saved.")
            else:
                logging.warning(f"Label {label} not found in label names dictionary and will be skipped.")


def run_multi_model_prediction(image_paths: List[str], models: List[dict], output: str, 
                               df: pd.DataFrame, multilabel: bool = False, force_split: bool = True,
                                crop_trunk: bool = False, name: Optional[str] = None, reorient: bool = True, 
                                largest_cc: bool = True, repair: bool = True):
    """
    Perform multiple predictions on a set of images using a list of models.
    
    Parameters:
        image_paths (list): List of paths to the input images.
        models (list): List of models to be used for prediction.
        output (str): Directory where the output segmentations will be saved.
        df (DataFrame): DataFrame containing additional information required for processing.
        multilabel (bool): Flag to enable multilabel mode. Default is False.
        force_split (bool): Flag to force splitting of the data. Default is True.
        crop_trunk (bool): Flag to enable cropping to the trunk region. Default is False.
        name (str, optional): Optional name for saving the segmentation. Default is None.
    """
    
    # Add file exists check 
    image_base = os.path.basename(image_paths[0]).split('.')[0]
    filename = f"{name}.nii.gz"
    output_name = os.path.join(output, image_base, f'{image_base}_{filename}')
    
    # --> This was pulled into the main can be depreciated
    if os.path.exists(output_name):
        print(f"{output_name} already exists. Skipping processing.")
        return 0
    else: 
        print(f"Processing: {output_name}")



    images = [read_image(image_path) for image_path in image_paths]

    if crop_trunk: 
        original_image = images[0]
        roi_size, roi_start = crop_image_to_trunk(original_image)
        images = [sitk.RegionOfInterest(im, size=roi_size, index=roi_start) for im in images]
    

    combined_segmentation = None
    label_names = {}
    for model in models:
        model_base = os.path.basename(model['path']).split('_')[0]
        model_path = os.path.join(model['path'], "__".join([model['trainer'], 'nnUNetPlans', model['config']]))
        logging.info(f"Running Model {model['path']}")
        logging.debug(f"Configuration {model}")
        processed_segments = process_image_with_model(images, model, df, force_split=force_split)

        for binary_segmentation, label in processed_segments:
            if crop_trunk:
                binary_segmentation = fill_cropped_image_into_original(binary_segmentation, original_image, roi_size, roi_start)
            
            label_name = fetch_label_name_from_json(model_path, label)
            label_names[label] = label_name

            logging.debug(f"Fetched label name: {label_name} for label: {label}")

            if combined_segmentation is None:
                combined_segmentation = binary_segmentation
                
            else:
                mask = sitk.Equal(combined_segmentation, 0)
                masked_segmentation = sitk.Mask(binary_segmentation, mask)
                combined_segmentation = sitk.Add(combined_segmentation, masked_segmentation)

    if largest_cc and repair:
        logging.warning('Exclusive option: Choose either --cc or --repair option')
    elif largest_cc:
        combined_segmentation = extract_largest_connected_component_multilabel(combined_segmentation)
    elif repair:
        combined_segmentation = repair_multilabel(combined_segmentation, 3)
    
    combined_segmentation = remove_small_components_multilabel(combined_segmentation, 250) # 125-500 Random threshold
    save_segmentation_result(output, image_base, model_base, combined_segmentation, label_names, multilabel, name=name)


def fill_mask_via_dilation(relabeled_mask, original_mask, max_iterations=15):
    logging.info('Starting the dilation process.')
    original_array = sitk.GetArrayFromImage(original_mask)
    relabeled_array = sitk.GetArrayFromImage(relabeled_mask)

    allowable_area = (original_array > 0) & (relabeled_array == 0)
    logging.debug(f'Initial allowable area size: {np.sum(allowable_area)}')

    labels = np.unique(relabeled_array)[1:]  # Exclude background
    if not labels.size:
        logging.warning("No labels found for dilation. Exiting.")
        return relabeled_mask

    iteration = 0
    any_change = True

    while any_change and iteration < max_iterations:
        logging.debug(f'Iteration {iteration + 1}')
        any_change = False
        updated_mask = np.copy(relabeled_array)

        for label in labels:
            current_label_mask = (relabeled_array == label)
            dilated_label_mask = binary_dilation(current_label_mask, structure=np.ones((3,3,3)))

            new_dilation = dilated_label_mask & allowable_area
            changes = np.sum(new_dilation != (current_label_mask & allowable_area))
            logging.debug(f'Attempting to fill {changes} pixels for label {label}')

            if changes > 0:
                updated_mask[new_dilation] = label
                any_change = True

        relabeled_array = updated_mask
        allowable_area = (original_array > 0) & (relabeled_array == 0)
        logging.debug(f'Updated allowable area size: {np.sum(allowable_area)}')

        iteration += 1

    output_image = sitk.GetImageFromArray(relabeled_array)
    output_image.CopyInformation(relabeled_mask)
    logging.info('Dilation process completed.')
    return output_image


def run_relabel(image_paths: List[str], models: List[dict], output: str, 
                               df: pd.DataFrame, label_image_pos=1, multilabel: bool = False, force_split: bool = True,
                                name: Optional[str] = None, reorient: bool = True, 
                                largest_cc: bool = True, repair: bool = True):
    """
    Perform multiple predictions on a set of images using a list of models.
    
    Parameters:
        image_paths (list): List of paths to the input images. Give Segmentation FIRST
        models (list): List of models to be used for prediction.
        output (str): Directory where the output segmentations will be saved.
        df (DataFrame): DataFrame containing additional information required for processing.
        multilabel (bool): Flag to enable multilabel mode. Default is False.
        force_split (bool): Flag to force splitting of the data. Default is True.
        crop_trunk (bool): Flag to enable cropping to the trunk region. Default is False.
        name (str, optional): Optional name for saving the segmentation. Default is None.
    """
    

    images = [read_image(image_path) for image_path in image_paths]
    original_image=images[label_image_pos]
    max_label = len(np.unique(sitk.GetArrayFromImage(images[label_image_pos]))[1:])
    logging.info(f"Found {max_label} for relabeling")
    all_labels_combined = None

    for label in np.unique(sitk.GetArrayFromImage(images[label_image_pos]))[1:]:    
        logging.info(f"Running relabelling model for label #: {label}")
        combined_segmentation = None

        roi_size, roi_start = crop_image_to_label(images[label_image_pos], label)
        
        cropped_images = [sitk.RegionOfInterest(im, size=roi_size, index=roi_start) for im in images]
        binary_mask = cropped_images[label_image_pos]==label
        cropped_images[label_image_pos] = binary_mask

        image_base = os.path.basename(image_paths[0]).split('.')[0]
        label_names = {}
        for model in models:
            model_base = os.path.basename(model['path']).split('_')[0]
            model_path = os.path.join(model['path'], "__".join([model['trainer'], 'nnUNetPlans', model['config']]))
            logging.info(f"Running Model {model['path']}")
            logging.debug(f"Configuration {model}")
            processed_segments = process_image_with_model(cropped_images, model, df, force_split=force_split)

            for binary_segmentation, label in processed_segments:
                binary_segmentation = fill_cropped_image_into_original(binary_segmentation, original_image, roi_size, roi_start)
                
                label_name = fetch_label_name_from_json(model_path, label)
                label_names[label] = label_name

                logging.debug(f"Fetched label name: {label_name} for label: {label}")

                if combined_segmentation is None:
                    combined_segmentation = binary_segmentation      
                else:
                    mask = sitk.Equal(combined_segmentation, 0)
                    masked_segmentation = sitk.Mask(binary_segmentation, mask)
                    combined_segmentation = sitk.Add(sitk.Cast(combined_segmentation, sitk.sitkInt16), sitk.Cast(masked_segmentation, sitk.sitkInt16) )

                if largest_cc and repair:
                    logging.warning('Exclusive option: Choose either --cc or --repair option')
                elif largest_cc:
                    logging.info('Starting largest_CC...')
                    combined_segmentation = extract_largest_connected_component_multilabel(combined_segmentation)
                elif repair:
                    logging.info('Starting repair...')
                    combined_segmentation = repair_multilabel(combined_segmentation, 3)

            if all_labels_combined is None:
                all_labels_combined = combined_segmentation
            else:
                mask = sitk.Equal(all_labels_combined, 0)
                masked_segmentation = sitk.Mask(combined_segmentation, mask)
                all_labels_combined = sitk.Add(sitk.Cast(all_labels_combined, sitk.sitkInt16), sitk.Cast(masked_segmentation, sitk.sitkInt16) )



    all_labels_combined = remove_small_components_multilabel(all_labels_combined, 125) # 125-500 Random threshold
    all_labels_combined = fill_mask_via_dilation(all_labels_combined, original_image, max_iterations=15)
    save_segmentation_result(output, image_base, model_base, all_labels_combined, label_names, multilabel, name=name)

def filter_existing_files(image_sets, output_dir, name):
    """
    Helper function to filter out image sets whose outputs already exist.
    """
    filtered_sets = []
    for image_set in image_sets:
        image_base = os.path.basename(image_set[0]).split('.')[0]
        filename = f"{name}.nii.gz"
        output_name = os.path.join(output_dir, image_base, f'{image_base}_{filename}')
        if not os.path.exists(output_name):
            filtered_sets.append(image_set)
        else:
            logging.info(f"{output_name} already exists. Skipping.")
    return filtered_sets

def main():
    """
    Main function to run the image processing pipeline.
    """
    global pool  # Allow signal handler to access the pool
    try:
        # Set multiprocessing start method to 'spawn' for CUDA compatibility
        if get_device() == "cuda":
            multiprocessing.set_start_method('spawn', force=True)

        configure_environment_variables()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # Handle `kill -TERM`

        parser = argparse.ArgumentParser(description='Process some images.')
        parser.add_argument('cmd', help='Command file name')
        parser.add_argument('--image', required=True, nargs='+', help='Path(s) to the input image(s). Multiple images only for multimodal predictions. Supports glob patterns.')
        parser.add_argument('--output', required=True, help='Path to the output directory')
        parser.add_argument('--ml', action='store_true', help='Enable multilabel mode')
        parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
        parser.add_argument('--split', action='store_true', help='Enable splitting mode')
        parser.add_argument('--trunk', action='store_true', help='Enable trunk cropping mode')
        parser.add_argument('--preserve', action='store_true', help='Enable no reorienting of images')
        parser.add_argument('--cc', action='store_true', help='Enable to keep only largest connected component label')
        parser.add_argument('--repair', action='store_true', help='Enable repair mode (if more than 1 CC)')
        parser.add_argument('--relabel', action='store_true', help='Enable repair mode (if more than 1 CC)')
        parser.add_argument('--mp', type=int, default=0, help='Number of worker processes for parallel processing (0 for no multiprocessing)')

        args = parser.parse_args()

        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

        logger = logging.getLogger()
        logging_level = logging.getLevelName(logger.getEffectiveLevel())
        logging.info(f"Current logging level: {logging_level}")

        # Expand and sort images for multimodal pairing
        multimodal_images = []
        for pattern in args.image:
            matched_files = sorted(glob(pattern))
            if not matched_files:
                logging.error(f"No matching files found for pattern: {pattern}")
                return
            multimodal_images.append(matched_files)

        # Determine single-label or multimodal
        if len(multimodal_images) == 1:
            paired_images = [(image,) for image in multimodal_images[0]]
        else:
            if not all(len(modality) == len(multimodal_images[0]) for modality in multimodal_images):
                logging.error("The number of images in each modality does not match. Ensure all glob patterns produce the same number of files.")
                return
            paired_images = list(zip(*multimodal_images))

        name = os.path.basename(args.cmd) if args.ml else None
        paired_images = filter_existing_files(paired_images, args.output, name)

        if not paired_images:
            logging.info("No new images to process. Exiting.")
            return

        logging.info(f"Images to process: {len(paired_images)}")

        models, df = load_models_from_json(args.cmd)

        name = None
        if args.ml:
            name = os.path.basename(args.cmd)

        logging.info(f"Running on {get_device()}...")

        # Setup multiprocessing if enabled
        if args.mp > 0:
            logging.info(f"Initializing multiprocessing pool with {args.mp} workers")
            pool = multiprocessing.Pool(processes=args.mp)

        if args.relabel:
            process_func = partial(process_image_set_relabel, models=models, output_dir=args.output, df=df,
                                   multilabel=args.ml, force_split=args.split, name=name,
                                   reorient=not args.preserve, largest_cc=args.cc, repair=args.repair)
        else:
            process_func = partial(process_image_set_prediction, models=models, output_dir=args.output, df=df,
                                   multilabel=args.ml, force_split=args.split, crop_trunk=args.trunk,
                                   name=name, reorient=not args.preserve, largest_cc=args.cc, repair=args.repair)

        if pool:
            pool.map(process_func, paired_images)
        else:
            for image_set in paired_images:
                try:
                    logging.info(f"Processing image set: {image_set}")
                    if args.relabel:
                        run_relabel(list(image_set), models, args.output, df, multilabel=args.ml,
                                    force_split=args.split, name=name, reorient=not args.preserve,
                                    largest_cc=args.cc, repair=args.repair)
                    else:
                        run_multi_model_prediction(list(image_set), models, args.output, df,
                                                    multilabel=args.ml, force_split=args.split,
                                                    crop_trunk=args.trunk, name=name,
                                                    reorient=not args.preserve, largest_cc=args.cc,
                                                    repair=args.repair)
                except Exception as e:
                    logging.error(f"Failed: {image_set} with {e}")
    finally:
        if pool:
            pool.close()
            pool.join()
        print("All resources cleaned up. Exiting.")

def process_image_set_prediction(image_set, models, output_dir, df, **kwargs):
    """Helper function for multiprocessing prediction tasks."""
    try:
        logging.info(f"Processing image set: {image_set}")
        run_multi_model_prediction(list(image_set), models, output_dir, df, **kwargs)
    except Exception as e:
        logging.error(f'Failed: {image_set} with {e}')

def process_image_set_relabel(image_set, models, output_dir, df, **kwargs):
    """Helper function for multiprocessing relabel tasks."""
    try:
        logging.info(f"Processing image set: {image_set}")
        run_relabel(list(image_set), models, output_dir, df, **kwargs)
    except Exception as e:
        logging.error(f'Failed: {image_set} with {e}')

  
if __name__ == "__main__":
    """
    Example command:
    nnunetmgr bonelabvertebra --image "/path/to/images/*_0000.nii.gz" --output /path/to/output --ml --verbose --trunk
    """
    main()