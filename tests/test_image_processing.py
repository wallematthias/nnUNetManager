import pytest
import os
import SimpleITK as sitk
import numpy as np
import json
import pandas as pd
import logging
import warnings
from unittest.mock import patch, mock_open
import tempfile

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="Please import.*")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from nnUNetManager.nnunetmgr import (
    get_device,
    extract_spacing_from_plans,
    fetch_label_name_from_json,
    fetch_label_value_from_json,
    load_images,
    rescale_itk_image,
    split_itk_image,
    stitch_itk_image,
    extract_largest_connected_component,
    perform_prediction,
    predict_with_nnunet,
    crop_image_to_trunk,
    fill_cropped_image_into_original,
    load_models_from_json,
    process_image_with_model,
    save_segmentation_result,
    run_multi_model_prediction,
    configure_environment_variables
)

# Test get_device function
def test_get_device():
    device = get_device()
    assert device in ['cuda', 'cpu']

# Test extract_spacing_from_plans function
def test_extract_spacing_from_plans(tmpdir):
    # Create a temporary json file with spacing information
    data = {
        "configurations": {
            "test_config": {
                "spacing": [1.0, 2.0, 3.0]
            }
        }
    }
    config_dir = tmpdir.mkdir("test_config")
    config_path = os.path.join(config_dir, "plans.json")
    with open(config_path, 'w') as f:
        json.dump(data, f)
    
    spacing = extract_spacing_from_plans(config_path)
    assert spacing == [3.0, 2.0, 1.0]

# Test fetch_label_name_from_json function
def test_fetch_label_name_from_json(tmpdir):
    # Create a temporary json file with label information
    data = {
        "labels": {
            "background": 0,
            "label1": 1
        }
    }
    model_path = tmpdir.mkdir("model")
    json_path = os.path.join(model_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    label_name = fetch_label_name_from_json(model_path, 1)
    assert label_name == "label1"
    label_name = fetch_label_name_from_json(model_path, 2)
    assert label_name == "Unknown label"

# Test fetch_label_value_from_json function
def test_fetch_label_value_from_json(tmpdir):
    # Create a temporary json file with label information
    data = {
        "labels": {
            "background": 0,
            "label1": 1
        }
    }
    model_path = tmpdir.mkdir("model")
    json_path = os.path.join(model_path, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    label_value = fetch_label_value_from_json(model_path, "label1")
    assert label_value == 1
    label_value = fetch_label_value_from_json(model_path, "label2")
    assert label_value == -1

# Test load_images function
def test_load_images():
    # Create a dummy ITK image
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    images = [image]
    
    stacked_images, metadata = load_images(images)
    
    assert stacked_images.shape == (1, 10, 10, 10)
    assert 'sitk_stuff' in metadata
    assert 'spacing' in metadata

# Test rescale_itk_image function
def test_rescale_itk_image():
    # Create a dummy ITK image
    image = sitk.Image(10, 10, 10, sitk.sitkFloat32)
    image.SetSpacing((1.0, 1.0, 1.0))
    
    target_spacing = (0.5, 0.5, 0.5)
    rescaled_image = rescale_itk_image(image, target_spacing)
    
    assert rescaled_image.GetSize() == (20, 20, 20)
    assert rescaled_image.GetSpacing() == target_spacing

# Example function to create a temporary ITK image for testing
def create_temp_itk_image(size=(10, 10, 10), spacing=(1.0, 1.0, 1.0), pixel_type=sitk.sitkUInt8):
    image = sitk.Image(size, pixel_type)
    image.SetSpacing(spacing)
    image.SetOrigin((0.0, 0.0, 0.0))
    # Correct direction matrix for a 3D image
    direction = (1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0)
    image.SetDirection(direction)
    return image

# Test split_itk_image function
def test_split_itk_image():
    image = create_temp_itk_image()
    parts = split_itk_image(image, margin=1)
    assert len(parts) == 3
    assert parts[0].GetSize() == (10, 10, 4)  # Overlapping region included

# Test stitch_itk_image function
def test_stitch_itk_image():
    image = create_temp_itk_image(size=(150, 150, 150))
    parts = split_itk_image(image, margin=20)
    stitched_image = stitch_itk_image(parts, margin=20)
    
    assert stitched_image.GetSize() == (150, 150, 150)

# Test extract_largest_connected_component function
def test_extract_largest_connected_component():
    image = create_temp_itk_image(size=(10, 10, 10))
    image[5, 5, 5] = 1
    image[6, 5, 5] = 1
    image[5, 6, 5] = 1
    
    largest_cc = extract_largest_connected_component(image)
    assert largest_cc.GetSize() == (10, 10, 10)
    
    # Ensure the largest connected component is as expected
    largest_cc_array = sitk.GetArrayFromImage(largest_cc)
    assert np.sum(largest_cc_array) == 3  # There should be 3 non-zero voxels

# Test perform_prediction function
# This would require a trained nnUNet model, so it's typically tested in an integration test setup

# Test predict_with_nnunet function
# This would require a trained nnUNet model, so it's typically tested in an integration test setup

# Test crop_image_to_trunk function
# This would require a trained nnUNet model, so it's typically tested in an integration test setup

# Test fill_cropped_image_into_original function
def test_fill_cropped_image_into_original():
    original_image = create_temp_itk_image()
    cropped_image = create_temp_itk_image(size=(5, 5, 5))
    roi_size = (5, 5, 5)
    roi_start = (2, 2, 2)
    filled_image = fill_cropped_image_into_original(cropped_image, original_image, roi_size, roi_start)
    assert filled_image.GetSize() == original_image.GetSize()

# Test load_models_from_json function
@patch.dict(os.environ, {"NNUNET_MODELS_DIR": "/fake/models", "NNUNET_CMD_DIR": "/fake/cmd"})
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
    "dataset1": {
        "order": 1,
        "trainer": "trainer1",
        "fold": 0,
        "checkpoint": "checkpoint",
        "config": "config",
        "labels": {"label1": 1, "label2": 2}
    }
}))

@pytest.fixture
def setup_env(tmpdir):
    # Create the necessary directories
    cmd_dir = tmpdir.mkdir("cmd")
    models_dir = tmpdir.mkdir("models")

    # Set the environment variables
    os.environ['NNUNET_MODELS_DIR'] = str(models_dir)
    os.environ['NNUNET_CMD_DIR'] = str(cmd_dir)

    return cmd_dir, models_dir

def test_load_models_from_json(setup_env):
    cmd_dir, models_dir = setup_env
    json_file = "test_cmd"
    json_path = os.path.join(cmd_dir, f"{json_file}.json")

    # Create the directory structure for the model
    model_dataset_dir = os.path.join(models_dir, "dataset1")
    os.makedirs(model_dataset_dir, exist_ok=True)
    trainer_config_dir = os.path.join(model_dataset_dir, "trainer1__nnUNetPlans__config")
    os.makedirs(trainer_config_dir, exist_ok=True)

    # Write the JSON content to the cmd directory
    with open(json_path, 'w') as f:
        json.dump({
            "dataset1": {
                "order": 1,
                "trainer": "trainer1",
                "fold": 0,
                "checkpoint": "checkpoint",
                "config": "config",
                "labels": {"label1": 1, "label2": 2}
            }
        }, f)

    # Write the necessary content to the dataset.json file
    dataset_json_content = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4
        },
        "numTraining": 210,
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    dataset_json_path = os.path.join(trainer_config_dir, "dataset.json")
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json_content, f)

    # Call the function to test
    models, df = load_models_from_json(json_file)

    # Assertions to verify the correctness of the models and df
    assert len(models) == 1
    assert 'path' in models[0]
    assert 'trainer' in models[0]
    assert 'config' in models[0]
    assert 'fold' in models[0]
    assert 'checkpoint_name' in models[0]

    assert not df.empty
    assert 'ORDER' in df.columns
    assert 'DATASET' in df.columns
    assert 'TRAINER' in df.columns
    assert 'FOLD' in df.columns
    assert 'CHECKPOINT' in df.columns
    assert 'CONFIG' in df.columns
    assert 'LABELNAME' in df.columns
    assert 'LABEL' in df.columns
    assert 'MULTILABEL' in df.columns
if __name__ == "__main__":
    pytest.main()
