import dataclasses
from dataclasses import dataclass
from dataclasses import fields
from typing import Optional
from typing import Union
from dataclass_wizard import YAMLWizard
from yaml.constructor import ConstructorError


@dataclass
class Config(YAMLWizard):
    """All parameters necessary to run the distance explainer."""
    experiment_name: str
    mask_selection_range_min: float
    mask_selection_range_max: float
    mask_selection_negative_range_min: float
    mask_selection_negative_range_max: float
    number_of_masks: Union[str, int]
    p_keep: Optional[float]
    feature_res: int
    random_seed: int
    manual_central_value: Optional[float]

    @classmethod
    def load(cls, path):
        """Load a config."""
        try:
            return cls.from_yaml_file(path)
        except ConstructorError:
            with open(path, 'r') as f:
                return cls.from_yaml('\n'.join(f.read().split('\n')[1:]))

    def __xor__(self, other):
        """Compares two Config objects and returns a list of parameter names that differ between them."""
        differing_fields = []
        for field in fields(Config):
            if getattr(self, field.name) != getattr(other, field.name):
                differing_fields.append(field.name)
        return differing_fields


original_config_options = Config(
    experiment_name='default',
    mask_selection_range_min=0,  # 0-1
    mask_selection_range_max=0.1,  # 0-1
    mask_selection_negative_range_min=0.9,  # 0-1
    mask_selection_negative_range_max=1,  # 0-1
    number_of_masks=1000,  # auto, [1, -> ]
    p_keep=0.5,  # None (auto), 0 - 1
    feature_res=8,  # [1, ->]
    random_seed=0,
    manual_central_value=0,
)


def get_default_config() -> Config:
    """Creates a distance explainer config with default values."""
    return dataclasses.replace(original_config_options)
