# Similar CXR Image Search

This command-line research tool ranks chest X-ray (CXR) images by the difference
between selected radiomics features. It uses PyRadiomics to extract features for
an image and its two lung regions, and reports the images with the smallest
feature differences. This is an experimental similarity search, not a diagnostic
tool.

## Requirements

Use Python with the packages in `requirements.txt`:

```shell
python -m pip install -r requirements.txt
```

Keras 2.2.4 also needs a compatible backend, such as TensorFlow, for image
segmentation. The backend is not pinned in `requirements.txt`.

## Files

- `cxr_search.py` — search and feature-extraction script.
- `trained_model.hdf5` — default lung-segmentation model.
- `4-mask.png` — default mask used when segmenting a query image; masks 1–3 are
  also included.
- `2.25.*.png` — sample CXR images.

The similarity index is **not included**. Provide your own gzip-compressed,
semicolon-delimited index file. Each row starts with an image identifier,
followed by numeric feature values in the order expected by the script.

## Usage

```text
python cxr_search.py --image IMAGE --index INDEX [--max MAX] [--mask MASK] [--model MODEL]
```

| Option | Description |
| --- | --- |
| `-i`, `--image` | Query image path. Supply this option; it is required in practice. |
| `-x`, `--index` | Required path to the gzip-compressed, semicolon-delimited feature index. |
| `-n`, `--max` | Maximum number of matches to print (default: `15`). |
| `-a`, `--mask` | Segmentation mask for a query image not found in the index (default: `4-mask.png`). |
| `-m`, `--model` | Segmentation model for a query image not found in the index (default: `trained_model.hdf5`). |

For example, from the project directory:

```shell
python cxr_search.py --image "2.25.65821805179691538693653550755504722887.png" --index "C:\data\depot_lung.csv.gz" --max 15
```

The result is a Python dictionary mapping image identifiers to distances,
ordered from smallest (most similar by the selected features) to largest.

## Query processing

The script first looks for the exact query-image path in the index. If found,
it reuses that row's features and does not load the mask or model. Otherwise, it
segments the query image with Keras and extracts radiomics features using the
provided mask and model. The default mask and model paths are relative to the
current working directory.

The score is the sum of absolute differences for the feature positions selected
in `cxr_search.py`; it is not a probability or a clinical similarity measure.
