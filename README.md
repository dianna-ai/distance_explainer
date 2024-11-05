[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10018768.svg)](https://doi.org/10.5281/zenodo.10018768) [![workflow pypi badge](https://img.shields.io/pypi/v/distance_explainer.svg?colorB=blue)](https://pypi.python.org/project/distance_explainer/)

# `distance_explainer`

XAI method to explain distances in embedded spaces.

![overview schema](https://github.com/user-attachments/assets/bbd5a79c-c50b-47a2-89fc-d8ed3053c845)


## Installation

To install distance_explainer from GitHub repository, do:

```console
git clone git@github.com:dianna-ai/distance_explainer.git
cd distance_explainer
python3 -m pip install .
```
## How to use

See our [tutorial](tutorial.ipynb) how to use this package.
In short:
```python
image1 = np.random.random((100, 100, 3))
image2 = np.random.random((100, 100, 3))

image2_embedded = model(image2)
explainer = DistanceExplainer(axis_labels={2: 'channels'})
attribution_map = explainer.explain_image_distance(model, image1, image2_embedded)
```
## Contributing

If you want to contribute to the development of distance_explainer,
have a look at the [contribution guidelines](docs/CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
