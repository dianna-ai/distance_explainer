[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10018768.svg)](https://doi.org/10.5281/zenodo.10018768) [![workflow pypi badge](https://img.shields.io/pypi/v/distance-explainer.svg?colorB=blue)](https://pypi.python.org/project/distance-explainer/) [![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dianna-ai.github.io/distance_explainer/)

# `distance_explainer`

XAI method to explain distances in embedded spaces.

![overview schema](https://raw.githubusercontent.com/dianna-ai/distance_explainer/main/docs/splash.png)


## Installation

There are 2 ways to install distance_explainer. To install distance_explainer from PyPI (recommended) run:

```console
pip install distance_explainer
```

To instead install distance_explainer from the GitHub repository, run:

```console
git clone git@github.com:dianna-ai/distance_explainer.git
cd distance_explainer
python3 -m pip install .
```
## How to use

See our [documentation](https://dianna-ai.github.io/distance_explainer/) and [tutorial](tutorial.ipynb) how to use this package.
In short:
```python
image1 = np.random.random((100, 100, 3))
image2 = np.random.random((100, 100, 3))

image2_embedded = model(image2)
explainer = DistanceExplainer(axis_labels={2: 'channels'})
attribution_map = explainer.explain_image_distance(model, image1, image2_embedded)
```

## If you use, please cite

If you use Distance Explainer for your research, please cite our method paper and the software itself:

- **Method paper:** "Explainable embeddings with Distance Explainer" — [arXiv:2505.15516](https://arxiv.org/abs/2505.15516) (to appear in the XAI26 proceedings)
- **Software:** `distance_explainer` — [doi:10.5281/zenodo.10018768](https://doi.org/10.5281/zenodo.10018768)

```bibtex
@article{meijer2025explainable,
  title={Explainable embeddings with {D}istance {E}xplainer},
  author={Meijer, Christiaan and Bos, E. G. Patrick},
  journal={arXiv preprint arXiv:2505.15516},
  year={2025},
  note={To appear in the XAI26 proceedings}
}

@software{meijer2023distance,
  title={distance_explainer},
  author={Meijer, Christiaan and Bos, Patrick},
  doi={10.5281/zenodo.10018768},
  year={2023}
}
```

## Contributing

If you want to contribute to the development of distance_explainer,
have a look at the [contribution guidelines](docs/CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
