# nnUNetManager

![Python Package](https://github.com/wallematthias/nnUNetManager/actions/workflows/python-package.yml/badge.svg) ![PyPi](https://github.com/wallematthias/nnUNetManager/actions/workflows/python-publish.yml/badge.svg)


nnUNetManager is a command-line tool designed to manage and execute nnUNet models for medical image segmentation tasks. It simplifies the execution of predefined commands stored in JSON files, facilitating complex segmentation workflows consisting out of several nnUNet models. Some functionality was inspired by [Totalsegmentator](https://github.com/wasserth/TotalSegmentator), however with a customizable workflow to add new models. 

All credit to the original developers of [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

## Installation

To install nnUNetManager, use pip:
```
git clone https://github.com/wallematthias/nnUNetManager.git
pip install -e .
```

## Download Resources

To download pre-trained models from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator). At this point, I would like to highlight that you should use their repository if you are only interested in the models they have trained. A good use case for this repository is when you want to combine several of your own models with each other or with TotalSegmentator. Thank you Jakob Wasserthal for training these. 
```
nnunetmgr-download

```

## Usage

nnUNetManager operates through command-line commands defined in JSON files located in the commands directory. Each command specifies a pretrained model and its configurations for segmentation tasks.

### Example Command File

Here's a generic example of a command file (commands/example_command.json):
```
{
    "dataset1_model": {
        "order": "1",
        "trainer": "nnUNetTrainerNoMirroring",
        "fold": "0",
        "checkpoint": "checkpoint_final",
        "config": "3d_fullres",
        "labels": {
            "label1": 1,
            "label2": 2,
            "label3": 3
        }
    },
    "dataset2_model": {
        "order": "2",
        "trainer": "nnUNetTrainerNoMirroring",
        "fold": "0",
        "checkpoint": "checkpoint_final",
        "config": "3d_fullres",
        "labels": {
            "labelA": 1,
            "labelB": 2,
            "labelC": 3
        }
    }
}
```
### Command Line Options

nnUNetManager supports several command line options:

- nnunetmgr <command>: Executes the specified command from the commands directory.
- --image path-to.nii.gz: Path to the input image(s). Multiple images are supported for multimodal predictions.
- --output path-to-output-folder: Path to the output directory where segmentation results will be saved.
- --ml: Enables multilabel mode.
- --verbose: Enables verbose mode for detailed logging.
- --split: Enables splitting mode.
- --trunk: Enables trunk cropping mode (requires downloading the pretrained dataset).
- --preserve: Disables reorienting of images.
- --cc: Only keeps the largest connected component of each label
- --repair: Attempts to reassign unconnected components to their neighbours. 

## Folder Structure

Ensure your models are organized in the standard output structure from nnUNet:

```
models/
|-- Dataset00X_XXXX/
    |-- dataset_fingerprint.json
    |-- dataset.json
    |-- plans.json
    |-- fold_X/
        |-- checkpoint_final.pth
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Special thanks to the nnUNet team for their contribution to medical image segmentation research.



