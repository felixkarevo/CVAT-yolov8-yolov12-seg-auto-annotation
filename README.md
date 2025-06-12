# CVAT-custom-yolov8-segmentation-auto-annotation

This GitHub repository contains the files to use custom YoloV8 and YoloV12 segmentation models serverless with Nuclio on CVAT with GPU support.

First, you need to set up all the serverless functionalities, install Nuclio as a function in Nuclio, etc. See the documentation [here](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/). The files in this repository can be used to deploy the Nuclio functions that run the custom YoloV8 or YoloV12 segmentation models. The files will create a CVAT Nuclio project to contain the functions within a Docker container. Commands should be run only after CVAT has been installed using Docker Compose because it runs the Nuclio dashboard which manages all serverless functions.

## Model Support

- **YoloV8 segmentation**: Original implementation for YoloV8 segmentation models
- **YoloV12 segmentation**: New implementation in the `/yolov12-seg` folder supporting the latest YoloV12 models

Besides the custom YoloV8/YoloV12 segmentation model, you need the `function-gpu.yaml` file for setting up Nuclio and a `main.py` script that runs in Nuclio and handles the data from the model.

## Installation Notes

**Flash Attention Dependency**: The YoloV12 implementation requires flash attention. The installation URL/wheel file provided in the configuration may need to be adjusted based on your specific system requirements, CUDA version, and Python version. You may need to manually update the flash attention installation command in the `function-gpu.yaml` file to match your environment. Check the [flash attention releases](https://github.com/Dao-AILab/flash-attention/releases) for the appropriate wheel file for your setup.

## Deployment

Once everything is set up, navigate to your CVAT folder from which you can deploy the Nuclio function. If you have a GPU, you can do it like this:

For YoloV8:
```bash
./serverless/deploy_gpu.sh ~/cvat/serverless/pytorch/ultralytics/custom_yolov8_GPU
```

For YoloV12:
```bash
./serverless/deploy_gpu.sh ~/path/to/yolov12-seg
```

I believe the scripts should also work if you don't have a GPU.

## Workflow

I use my custom yolov8/yolov12 segmentation model to (bulk) auto annotate all images. However, I find that it is sometimes not accurate enough. Therefore, as a second step I am using segment anything to correct labels or make them more fine-grained. I find this two step approach to generate very accurate labels. 
