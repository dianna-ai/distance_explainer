"""Documentation about distance_explainer."""

import logging
import dianna.utils
import numpy as np
from dianna.utils.maskers import generate_masks_for_images
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Christiaan Meijer"
__email__ = "c.meijer@esciencecenter.nl"
__version__ = "0.2.0"


class DistanceExplainer:
    """Explainer object to explain an image with respect to a reference point in an embedded space."""
    # axis labels required to be present in input image data
    required_labels = ('channels',)

    def __init__(self, n_masks=1000, feature_res=8, p_keep=.5,  # pylint: disable=too-many-arguments
                 mask_selection_range_max=0.2, mask_selection_range_min=0, mask_selection_negative_range_max=1,
                 mask_selection_negative_range_min=0.8, axis_labels=None, batch_size=10,
                 preprocess_function=None):
        """Creates an explainer object to explain an image with respect to a reference point in an embedded space.

        Args:
            n_masks: Number of masks to use to mask the input image. More increases reliability and computation.
            feature_res: Number of features per dimension in the image. Determines the super pixel size in the masks.
            p_keep: Probability of keeping features unmasked. Higher means less masked input.
            batch_size: Number of masked inputs to process in a single batch by the model
            axis_labels: Axis labels
            preprocess_function: Preprocess function
            mask_selection_range_max: Top end of range of outcomes that will be selected.
            mask_selection_range_min: Lower end of range of outcomes that will be selected.
            mask_selection_negative_range_max: Top end of range of outcomes that will be selected and weighted -1.
            mask_selection_negative_range_min: Lower end of range of outcomes that will be selected and weighted -1.
        """
        self.n_masks = n_masks
        self.feature_res = feature_res
        self.p_keep = p_keep
        self.preprocess_function = preprocess_function
        self.masks = None
        self.predictions = None
        self.axis_labels = axis_labels if axis_labels is not None else []
        self.mask_selection_range_max = mask_selection_range_max
        self.mask_selection_range_min = mask_selection_range_min
        self.mask_selection_negative_range_max = mask_selection_negative_range_max
        self.mask_selection_negative_range_min = mask_selection_negative_range_min
        self.batch_size = batch_size

    def explain_image_distance(self, model_or_function, input_data, embedded_reference, masks=None):
        """Explain an image with respect to a reference point in an embedded space.

        Args:
            model_or_function: Model that will encode the input_data into an embedded space
            input_data: Input data to be explained, by exploring what parts make it closer to the reference point.
            embedded_reference: Reference point in the embedded space
            masks: User specified masks, in case no autogenerated masks should be used.

        Returns:
            saliency map and the neutral value within the saliency map which indicates the parts of the image that
            neither bring the image closer nor further away from the embedded reference.
        """
        full_preprocess_function, input_data = self._prepare_input_data(input_data)
        runner = dianna.utils.get_function(model_or_function, preprocess_function=full_preprocess_function)
        active_p_keep = 0.5 if self.p_keep is None else self.p_keep  # Could autotune here (See #319)

        # data shape without batch axis and channel axis
        img_shape = input_data.shape[1:3]
        # Expose masks for to make user inspection possible
        if masks is None:
            self.masks = generate_masks_for_images(img_shape, self.n_masks, active_p_keep, self.feature_res)
        else:
            self.masks = masks
            if self.masks.shape[0] != self.n_masks:
                raise ValueError(f"Configured n_masks ({self.n_masks}) is not equal to the number of masks passed "
                                 f"({self.masks.shape[0]}).")

        # Make sure multiplication is being done for correct axes
        masked = input_data * self.masks

        batch_predictions = []

        for i in tqdm(range(0, self.n_masks, self.batch_size), desc='Explaining'):
            new_predictions = runner(masked[i:i + self.batch_size])
            batch_predictions.append(new_predictions)

        self.predictions = np.concatenate(batch_predictions)

        def describe(x, name):
            return f'Description of {name}\nmean:{np.mean(x)}\nstd:{np.std(x)}\nmin:{np.min(x)}\nmax:{np.max(x)}'

        statistics = []

        highest_distances_masks, highest_mask_weights = self._get_lowest_distance_masks_and_weights(
            embedded_reference,
            self.predictions, self.masks,
            self.mask_selection_negative_range_min,
            self.mask_selection_negative_range_max)

        if len(highest_mask_weights) > 0:
            statistics.append(describe(highest_mask_weights, 'highest_mask_weights'))
            unnormalized_sal_highest = np.mean(highest_distances_masks, axis=0)
        else:
            unnormalized_sal_highest = 0

        lowest_distances_masks, lowest_mask_weights = self._get_lowest_distance_masks_and_weights(
            embedded_reference,
            self.predictions, self.masks,
            self.mask_selection_range_min,
            self.mask_selection_range_max)

        if len(lowest_mask_weights) > 0:
            statistics.append(describe(lowest_mask_weights, 'lowest_mask_weights'))
            unnormalized_sal_lowest = np.mean(lowest_distances_masks, axis=0)
        else:
            unnormalized_sal_lowest = 0

        self.statistics = '\n'.join(statistics)
        unnormalized_sal = unnormalized_sal_lowest - unnormalized_sal_highest

        saliency = unnormalized_sal

        neutral_value = active_p_keep

        # for one-sided experiments, use "meaningful" neutral value (the unperturbed distance), otherwise center on 0
        if len(lowest_mask_weights) > 0 and len(highest_mask_weights) == 0:
            neutral_value = neutral_value
        if len(highest_mask_weights) > 0 and len(lowest_mask_weights) == 0:
            neutral_value = -neutral_value
        if len(highest_mask_weights) > 0 and len(lowest_mask_weights) > 0:
            neutral_value = 0

        return saliency, neutral_value

    @staticmethod
    def _get_lowest_distance_masks_and_weights(embedded_reference, predictions, masks, mask_selection_range_min,
                                               mask_selection_range_max):
        distances = DistanceExplainer.calculate_distances(predictions, embedded_reference)
        lowest_distances_indices = np.argsort(distances, axis=0)[
                                   int(len(predictions) * mask_selection_range_min)
                                   :int(len(predictions) * mask_selection_range_max)]
        mask_weights = np.exp(-distances[lowest_distances_indices])
        lowest_distances_masks = masks[lowest_distances_indices]
        return lowest_distances_masks, mask_weights

    @staticmethod
    def calculate_distances(predictions: np.ndarray, embedded_reference: np.ndarray) -> np.ndarray:
        """Calculate the distances to the reference point in an embedded space using cosine distance with range [0,1].

        Args:
            predictions: Batch of points in the embedded space for with distances are calculated
            embedded_reference: Point to calculate the distance to

        Returns:
            Distances from each point to the reference point
        """
        distances = pairwise_distances(predictions, embedded_reference,
                                       metric='cosine') / 2  # divide by 2 to have [0.1] output range
        return distances

    def _prepare_input_data(self, input_data):
        input_data_xarray = dianna.utils.to_xarray(input_data, self.axis_labels, DistanceExplainer.required_labels)
        input_data_xarray_expanded = input_data_xarray.expand_dims('batch', 0)
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data_xarray_expanded.dims.index('channels')
        prepared_input_data = dianna.utils.move_axis(input_data_xarray_expanded, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, prepared_input_data.dtype)
        return full_preprocess_function, prepared_input_data

    def _prepare_image_data(self, input_data):
        """Transforms the data to be of the shape and type RISE expects.

        Args:
            input_data (xarray): Data to be explained

        Returns:
            transformed input data, preprocessing function to use with dianna.utils.get_function()
        """
        # ensure channels axis is last and keep track of where it was so we can move it back
        channels_axis_index = input_data.dims.index('channels')
        input_data = dianna.utils.move_axis(input_data, 'channels', -1)
        # create preprocessing function that puts model input generated by RISE into the right shape and dtype,
        # followed by running the user's preprocessing function
        full_preprocess_function = self._get_full_preprocess_function(channels_axis_index, input_data.dtype)
        return input_data, full_preprocess_function

    def _get_full_preprocess_function(self, channel_axis_index, dtype):
        """Creates a full preprocessing function.

        Creates a preprocessing function that incorporates both the (optional) user's
        preprocessing function, as well as any needed dtype and shape conversions

        Args:
            channel_axis_index (int): Axis index of the channels in the input data
            dtype (type): Data type of the input data (e.g. np.float32)

        Returns:
            Function that first ensures the data has the same shape and type as the input data,
            then runs the users' preprocessing function
        """

        def moveaxis_function(data):
            return dianna.utils.move_axis(data, 'channels', channel_axis_index).astype(dtype).values

        if self.preprocess_function is None:
            return moveaxis_function
        return lambda data: self.preprocess_function(moveaxis_function(data))
