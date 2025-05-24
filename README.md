# Similar CXR Image Search

## Synopsis
This script search for similar CXR image based on radiomics features generated using the pyradiomics
packages (https://pyradiomics.readthedocs.io/en/latest/). The research returns 15 images with the highest
similarity. This script requires several libraries and modules such as `SimpleITK`, `scikit image`, and
`keras`. The search algorithm was designed based on the assumptions that image with similar attributes,
such as tuberculosis infection and cavity, will generate similar radiomics features. The goal of the
algorithm is then to find the images with most similar radiomics features as a proxy for finding images
with similar features.

## Requirements
- `SimpleITK`: Insight Segmentation and Registration Toolkit (ITK)
- `radiomics`: extraction of radiomics features from medical images
- `keras`: Python interface for artificial neural networks
- `skimage`: image processing modules for Python

## Usage
```
cxr_search.py [-h] [--image IMAGE] --index INDEX [--max MAX] [--mask MASK] [--model MODEL]

optional arguments:
  -h, --help                show this help message and exit
  --image IMAGE, -i IMAGE   [Required] Query image. (default: None)
  --index INDEX, -x INDEX   [Required] Similarity index file in CSV format. (default: None)
  --max MAX, -n MAX         [Optional] Maximum number of similar images. (default: 15)
  --mask MASK, -a MASK      [Optional] Lung segmentation mask. (default: 4-mask.png)
  --model MODEL, -m MODEL   [Optional] Lung segmentation model. (default: trained_model.hdf5)
```

Note: The script detects if the image is present in the similarity index file. If it does, the
script will simply retrieve the radiomics features from the index file, and then compares it with
all other images in the file. If the image is not present in the index file, the script will
trigger `keras` neural network and calculate the radiomics features of the image. The radiomics
features are then compare with all the images in the index file.

## Example
* Search with image in the index file:

`cxr_search.py -i 2.25.65821805179691538693653550755504722887.png -x depot_lung.csv.gz -n 15`

The script will generate a list of 15 most similar (least difference) images:

```
{
    '2.25.195285892709766271526784070875018613795.png': 0.27653489974943946,
    '2.25.300756047989880637409261393094152894584.png': 0.29335717637125325,
    '2.25.181953574516517966192159557388375242327.png': 0.31636982346929243,
    '2.25.24682019196261606431412262500199432179.png': 0.32959884257284433,
    '2.25.13129495731931299819594573266372711598.png': 0.3316487156514002,
    '2.25.16866037304459794827251756210100940012.png': 0.34999323055522313,
    '2.25.21889738079808477820625672908780007358.png': 0.36584268155522315,
    '2.25.84598080505877736515897475241228930376.png': 0.3742476505552231,
    '2.25.127752333903200436058273426221590001384.png': 0.3751025915552231,
    '2.25.245709938985681712298311679917888376274.png': 0.3756346565552231,
    '2.25.199923000469948579796895036848116821096.png': 0.38236147337125326,
    '2.25.52031954877009587164440253841591597788.png': 0.3830690385552231,
    '2.25.126325091911379364385182047950014931635.png': 0.3850874606514002,
    '2.25.174498520901772243936677903040006474706.png': 0.3884572888547224,
    '2.25.323006201739771781803093866916970267835.png': 0.3912392856532623
}
```

Note: The segmentation mask and neural network model are not required if the image is present in
the index file.

