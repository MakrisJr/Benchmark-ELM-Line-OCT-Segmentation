# ELM Line OCT Dataset
Benchmarking Automated Detection Of The Retinal External Limiting Membrane In A 3D Spectral Domain Optical Coherence Tomography Image Dataset of Full Thickness Macular Holes

## Prerequisites
+ Linux
+ Python with numpy
+ NVIDIA GPU + CUDA 10.0 
+ pytorch 1.10.0
+ torchvision

## Getting Started


**+ Clone this repo:**

    cd Benchmark-ELM-Line-OCT-Dataset

**+ Get dataset**

    Available on request from the corresponding author for noncommercial use.

**+ Train the model:**

    python train.py

**+ Test the model:**

    python predict.py

## Citation:
If you use the code in your work, please use the following citation:
```
@article{singh2021benchmarking,
  title={Benchmarking Automated Detection Of The Retinal External Limiting Membrane In A 3D Spectral Domain Optical Coherence Tomography Image Dataset of Full Thickness Macular Holes},
  author={Singh, VK and Kucukgoz, B and Murphy, DC and Xiong, X and Steel, DH and Obara, B},
  journal={Computers in Biology and Medicine},
  year={2021},
  publisher={Newcastle University}
}
```

### To create the environment:
`conda env create -f environment.yaml`

### File guide: 
- dataset.py: BasicDataset class, returns {image, mask} items
- dice_loss.py: implements dice coefficient and dice loss


## Utilities

- stack_to_tif.py: stack 2D mask images (e.g., from `result/`) into a 3D multi-page TIF named `<name>_seg.tif` compatible with `holes_detection.py` and `line_check.py`.

### Export a 3D _seg.tif from 2D predictions

After running `predict.py`, masks are saved as individual 2D images under `result/`. To analyze them with `holes_detection.py` or `line_check.py`, stack them into a multi-page TIF:

```
python stack_to_tif.py --input-dir result --output IMAGES/EBU_seg.tif
```

Notes:
- Images are sorted in natural numeric order to form the z-stack.
- Output volume shape is [Z, H, W]. If predictions were resized (e.g., 256x256), the stacked TIF will have that size.


- holes_detection.py: standalone utility, 


image 605-19 is missing, I have copied 605-18 as a temporary fix
same for 389-24 + 1 more 389 slice


## Results Test Set
SegNet: 0.665429
UNet: 0.671788
UNet3D: 0.727703
Frawley3D: 0.751104
UNet2DEnc3DDec: 0.737300
UNet3D_Aniso: 0.746722