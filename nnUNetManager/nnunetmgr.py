import os
import json
import logging
import argparse
import torch
import SimpleITK as sitk
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from batchgenerators.utilities.file_and_folder_operations import join
from skimage.measure import label
from matplotlib import pyplot as plt
from glob import glob
import random
import string
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from nibabel.orientations import io_orientation, inv_ornt_aff, apply_orientation
import requests
import zipfile



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def get_device() -> str:
    """Determine and return the available device for computation."""
    return 'cuda' if torch.cuda.is_available() else 'cpu'



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
    logging.debug(f"Largest label identified: {largest_label} with {label_shape_filter.GetNumberOfPixels(largest_label)} pixels.")

    largest_component = sitk.BinaryThreshold(connected_components, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1, outsideValue=0)
    return largest_component

def perform_prediction(images: List[sitk.Image], path: str, trainer: str, config: str, checkpoint_name: str = 'checkpoint_final.pth', fold: int = 0, target_spacing: Optional[Tuple[float, float, float]] = None, force_split: bool = True) -> sitk.Image:
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
    nr_voxels_thr = 512 * 512 * 900
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

def predict_with_nnunet(images: List[sitk.Image], model_path: str, checkpoint_name: str = 'checkpoint_final.pth', fold: int = 0) -> sitk.Image:
    """
    Predict output using a nnU-Net model for given images.
    
    Parameters:
        images (list): List of ITK images for prediction.
        model_path (str): Path to the trained nnUNet model directory.
        checkpoint_name (str): Checkpoint file name. Default is 'checkpoint_final.pth'.
        fold (int): Fold number. Default is 0.
        
    Returns:
        SimpleITK.Image: ITK image of the prediction.
    """
    logger = logging.getLogger()
    verbose = logger.getEffectiveLevel() == logging.DEBUG

    predictor = nnUNetPredictor(
        tile_step_size=0.8,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device(get_device()),
        verbose=verbose,
        verbose_preprocessing=verbose,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(model_path, use_folds=(fold,), checkpoint_name=checkpoint_name)

    img, props = load_images(images)

    #save_prediction_slice(images[0])


    prediction = predictor.predict_single_npy_array(img, props, None, None, False)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image.SetSpacing(images[0].GetSpacing())
    prediction_image.SetOrigin(images[0].GetOrigin())
    prediction_image.SetDirection(images[0].GetDirection())

    #save_prediction_slice(prediction_image)

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
        download_model(home)  # You need to implement this function to handle the download
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

def download_model(model_path: str):
    """
    Download a zip file from a URL, extract its contents into the specified path,
    and delete the zip file after extraction.

    Parameters:
        model_path (str): The path where the model should be extracted.
    """
    # URL of the zip file to download
    url = "https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/Dataset300_body_6mm_1559subj.zip"
    
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

    empty_img = sitk.Image(original_image.GetSize(), original_image.GetPixelID())
    empty_img.CopyInformation(original_image)

    paste_filter = sitk.PasteImageFilter()
    paste_filter.SetDestinationIndex(roi_start)
    paste_filter.SetSourceSize(cropped_image.GetSize())

    filled_image = paste_filter.Execute(empty_img, cropped_image)
    logging.debug(f"Restoring size from: {cropped_image.GetSize()} to: {filled_image.GetSize()}")
    return filled_image

def process_image_with_model(images: List[sitk.Image], model: dict, df: pd.DataFrame, multilabel: bool, force_split: bool = True) -> List[Tuple[sitk.Image, int]]:
    """
    Process an image using the specified model.
    
    Parameters:
        images (list): List of ITK images to process.
        model (dict): Model configuration.
        df (DataFrame): DataFrame with model information.
        multilabel (bool): Flag to enable multilabel mode.
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
            #binary_segmentation = extract_largest_connected_component(segmentation == label)
            binary_segmentation = segmentation == label
            if multilabel:
                multilabel_value = label_row['MULTILABEL'].values[0]
                binary_segmentation = sitk.Cast(binary_segmentation, sitk.sitkUInt16) * multilabel_value
            processed_segments.append((binary_segmentation, label))
    return processed_segments


def save_segmentation_result(output_path: str, image_base: str, model_base: str, segmentation: sitk.Image, label_name: str, multilabel: bool, properties_list: dict, name: Optional[str] = None):
    """
    Save the segmentation to the specified output path.
    
    Parameters:
        output_path (str): Path to the output directory.
        image_base (str): Base name of the image.
        model_base (str): Base name of the model.
        segmentation (SimpleITK.Image): Segmentation image.
        label_name (str): Label name.
        multilabel (bool): Flag to enable multilabel mode.
        name (str, optional): Optional name for saving the segmentation.
    """
    if multilabel:
        logging.info('Saving multilabel file')
        filename = f"{name}.nii.gz"
        output_name = os.path.join(output_path, image_base, filename)
    else:
        filename = f"{label_name}.nii.gz"
        output_name = os.path.join(output_path, image_base, model_base, filename)
    

    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    logging.info(f'Saving {output_name}')

    segmentation = sitk.Cast(segmentation, sitk.sitkUInt16)

    write_image(segmentation, output_name, properties_list)
    #sitk.WriteImage(segmentation, output_name)


def run_multi_model_prediction(image_paths: List[str], models: List[dict], output: str, 
                               df: pd.DataFrame, multilabel: bool = False, force_split: bool = True,
                                crop_trunk: bool = False, name: Optional[str] = None, reorient: bool = True):
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
    

    results = [read_image(image_path, reorient) for image_path in image_paths]
    images, properties_list = zip(*results)

    if crop_trunk: 
        original_image = images[0]
        roi_size, roi_start = crop_image_to_trunk(original_image)
        images = [sitk.RegionOfInterest(im, size=roi_size, index=roi_start) for im in images]
    
    image_base = os.path.basename(image_paths[0]).split('.')[0]
    combined_segmentation = None if multilabel else []

    for model in models:
        model_base = os.path.basename(model['path']).split('_')[0]
        model_path = os.path.join(model['path'], "__".join([model['trainer'], 'nnUNetPlans', model['config']]))
        logging.info(f"Running Model {model['path']}")
        logging.debug(f"Configuration {model}")
        processed_segments = process_image_with_model(images, model, df, multilabel, force_split=force_split)

        for binary_segmentation, label in processed_segments:
            if crop_trunk:
                processed_segments = fill_cropped_image_into_original(binary_segmentation, original_image, roi_size, roi_start)
            
            label_name = fetch_label_name_from_json(model_path, label)
            logging.debug(f"Fetched label name: {label_name} for label: {label}")

            if multilabel:
                if combined_segmentation is None:
                    combined_segmentation = binary_segmentation
                else:
                    mask = sitk.Equal(combined_segmentation, 0)
                    masked_segmentation = sitk.Mask(binary_segmentation, mask)
                    combined_segmentation = sitk.Add(combined_segmentation, masked_segmentation)
            else:
                save_segmentation_result(output, image_base, model_base, binary_segmentation, label_name, multilabel, properties_list[0], name=name)
    
    if multilabel:
        save_segmentation_result(output, image_base, model_base, combined_segmentation, None, multilabel, properties_list[0], name=name)


def read_image(file_path, reorient=True):
    """
    Read a NIfTI image, optionally reorient to RAS using nibabel, and convert it to a SimpleITK image.
    
    Parameters:
    file_path (str): Path to the NIfTI image file.
    reorient (bool): Whether to reorient the image to RAS. Default is True.
    
    Returns:
    SimpleITK.Image: The reoriented image as a SimpleITK image.
    dict: Properties including original and reoriented affines.
    """
    # Load the image using nibabel
    nib_image = nib.load(file_path)
    
    # Ensure the image is 3D
    assert nib_image.ndim == 3, 'Only 3D images are supported.'
    
    # Get the original affine
    original_affine = nib_image.affine

    if reorient:
        # Reorient the image to RAS
        orig_ornt = io_orientation(original_affine)
        target_ornt = [[0, 1], [1, 1], [2, 1]]
        transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
        reoriented_data = apply_orientation(nib_image.get_fdata(), transform)
        
        # Compute the reoriented affine
        reoriented_affine = np.dot(original_affine, inv_ornt_aff(transform, nib_image.shape))
    else:
        reoriented_data = nib_image.get_fdata()
        reoriented_affine = original_affine
    
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(reoriented_data.transpose(2, 1, 0))
    spacing = list(map(float, reversed(nib_image.header.get_zooms())))
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(tuple(reoriented_affine[:3, 3]))
    sitk_image.SetDirection(reoriented_affine[:3, :3].flatten().tolist())

    properties = {
        'original_affine': original_affine,
        'reoriented_affine': reoriented_affine,
        'reoriented': reorient
    }
    
    return sitk_image, properties

def write_image(sitk_image, file_path, properties):
    """
    Convert a SimpleITK image to a nibabel NIfTI image, optionally reorient back to original orientation, and save it.
    
    Parameters:
    sitk_image (SimpleITK.Image): The SimpleITK image to be saved.
    file_path (str): Path to save the NIfTI image file.
    properties (dict): Properties including original and reoriented affines.
    """
    original_affine = properties['original_affine']
    reoriented_affine = properties['reoriented_affine']
    reoriented = properties['reoriented']

    data = sitk.GetArrayFromImage(sitk_image).transpose(2, 1, 0)
    if reoriented:
        orig_ornt = io_orientation(original_affine)
        target_ornt = io_orientation(reoriented_affine)
        transform = nib.orientations.ornt_transform(target_ornt, orig_ornt)
        restored_data = apply_orientation(data, transform)
    else:
        restored_data = data

    nifti_image = nib.Nifti1Image(restored_data, original_affine)
    nib.save(nifti_image, file_path)

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


def main():
    """
    Main function to run the image processing pipeline.
    """
    configure_environment_variables()

    parser = argparse.ArgumentParser(description='Process some images.')
    parser.add_argument('cmd', help='Command file name')
    parser.add_argument('--image', required=True, nargs='+', help='Path(s) to the input image(s). Multiple images only for multimodal predictions')
    parser.add_argument('--output', required=True, help='Path to the output directory')
    parser.add_argument('--ml', action='store_true', help='Enable multilabel mode')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose mode')
    parser.add_argument('--split', action='store_true', help='Enable splitting mode')
    parser.add_argument('--trunk', action='store_true', help='Enable trunk cropping mode')
    parser.add_argument('--preserve', action='store_true', help='Enable no reorienting of images')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    logger = logging.getLogger()
    logging_level = logging.getLevelName(logger.getEffectiveLevel())
    logging.info(f"Current logging level: {logging_level}")
    
    models, df = load_models_from_json(args.cmd)
    
    name = None
    if args.ml:
        name = os.path.basename(args.cmd)
    
    logging.info(f"Running on {get_device()}...")

    run_multi_model_prediction(args.image, models, args.output, df, multilabel=args.ml, force_split=args.split, crop_trunk=args.trunk, name=name, reorient=args.preserve==False)


if __name__ == "__main__":
    """
    Example command:
    nnunetmgr bonelabvertebra --image /path/to/image.nii.gz --output /path/to/output --ml --verbose --trunk
    """
    main()
