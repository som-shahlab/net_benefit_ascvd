import torch
import torch.nn.functional as F
import itertools
import warnings
import logging

from net_benefit_ascvd.prediction_utils.pytorch_utils.models import (
    TorchModel,
    FixedWidthModel,
    BilevelModel,
)
from net_benefit_ascvd.prediction_utils.pytorch_utils.layers import FeedforwardNet
from net_benefit_ascvd.prediction_utils.pytorch_utils.pytorch_metrics import (
    weighted_mean,
    roc_auc_score_surrogate,
    precision_surrogate,
    tpr_surrogate,
    fpr_surrogate,
    positive_rate_surrogate,
    weighted_cross_entropy_loss,
    MetricUndefinedError,
    baselined_loss,
)

from net_benefit_ascvd.prediction_utils.pytorch_utils.pytorch_metrics import get_surrogate

# Structure:
# group_regularized_model -> function returning a model class based on a provided key
# GroupRegularizedModel -> upper level model class. Defines a regularized loss that can be computed arbitrarily
#   MMDModel -> penalizes MMD between model predictions for conditional prediction parity
#   EqualMeanPredictionModel -> penalizes mean difference in predictions with same interface as MMDModel
#   GroupIRMModel -> Applies the IRM penalty to each group
#   EqualThresholdRateModel -> penalizes thresholded prediction rates across groups with same interface as MMDModel
# GroupMetricRegularizedModel -> penalizes the differences in a differentiable metric across groups
#   EqualAUCModel
#   EqualPrecisionModel
#   EqualLossModel
#   EqualBrierScoreModel
#   EqualTPRModel
#   EqualFPRModel
#   EqualPositiveRateModel
# GroupAdversarialModel -> penalizes difference in distribution of predictions across groups w/ discriminator


def group_regularized_model(model_type="loss"):
    """
    A function that returns an instance of GroupRegularizedModel
    """
    class_dict = {
        "loss": EqualLossModel,
        "baselined_loss": EqualBaselinedLossModel,
        "auc": EqualAUCModel,
        "brier": EqualBrierScoreModel,
        "mmd": MMDModel,
        "mean_prediction": EqualMeanPredictionModel,
        "threshold_rate": EqualThresholdRateModel,
        "precision": EqualPrecisionModel,
        "adversarial": GroupAdversarialModel,
        "group_irm": GroupIRMModel,
    }
    the_class = class_dict.get(model_type, None)
    if the_class is None:
        raise ValueError("model_type not defined in group_regularized_model")
    return the_class


class GroupRegularizedModel(TorchModel):
    """
    A model that penalizes differences in a quantity across groups
    """

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Computes a regularization term defined in terms of model outputs, labels, and group.
        This class should be overriden and this regularization term defined.
        """
        raise NotImplementedError

    def get_default_config(self):
        """
        Default parameters
        """
        config_dict = super().get_default_config()
        update_dict = {
            "num_hidden": 1,
            "hidden_dim": 128,
            "drop_prob": 0.0,
            "normalize": False,
            "sparse": True,
            "sparse_mode": "csr",  # alternatively, "convert"
            "resnet": False,
        }
        return {**config_dict, **update_dict}

    def init_model(self):
        model = FeedforwardNet(
            in_features=self.config_dict["input_dim"],
            hidden_dim_list=self.config_dict["num_hidden"]
            * [self.config_dict["hidden_dim"]],
            output_dim=self.config_dict["output_dim"],
            drop_prob=self.config_dict["drop_prob"],
            normalize=self.config_dict["normalize"],
            sparse=self.config_dict["sparse"],
            sparse_mode=self.config_dict["sparse_mode"],
            resnet=self.config_dict["resnet"],
        )
        return model

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        result = super().get_transform_batch_keys()
        result = result + ["group"]

    def get_loss_names(self):
        return ["loss", "supervised", "group_regularization"]

    def forward_on_batch(self, the_data):
        """
        Run the forward pass, returning a batch_loss_dict and outputs
        """
        loss_dict_batch = {}
        inputs, labels, group = (
            the_data["features"],
            the_data["labels"],
            the_data["group"],
        )
        outputs = self.model(inputs)
        # Compute the loss
        if self.config_dict.get("weighted_loss"):
            loss_dict_batch["supervised"] = self.criterion(
                outputs, labels, sample_weight=the_data.get("weights")
            )
            loss_dict_batch[
                "group_regularization"
            ] = self.compute_group_regularization_loss(
                outputs, labels, group, sample_weight=the_data.get("weights")
            )
        else:
            loss_dict_batch["supervised"] = self.criterion(outputs, labels)
            loss_dict_batch[
                "group_regularization"
            ] = self.compute_group_regularization_loss(outputs, labels, group)

        loss_dict_batch["loss"] = loss_dict_batch["supervised"] + (
            self.config_dict["lambda_group_regularization"]
            * loss_dict_batch["group_regularization"]
        )
        return loss_dict_batch, outputs


class MMDModel(GroupRegularizedModel):
    """
    Model that minimizes distributional discrepancy between predictions belonging to different groups.
    In the default case, corresponds to threshold-free demographic parity.
    If made conditional on the outcome, corresponds to equalized odds.
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "mmd_mode": "conditional"
            # "conditional" -> eq_odds,
            # "conditional_pos" -> eq_opportunity_pos,
            # "conditional_neg" -> eq_opportunity_neg,
            # "unconditional" -> demographic_parity
        }
        return {**config_dict, **update_dict}

    def compute_mmd(self, x, y, x_weights=None, y_weights=None):
        """
        Compute an MMD
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)

        if (x_weights is None) and (y_weights is None):
            return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        else:
            if x_weights is None:
                x_weights = torch.ones(x.shape[0])
            if y_weights is None:
                y_weights = torch.ones(y.shape[0])

            x_weights_tile = x_weights.unsqueeze(1) * x_weights.unsqueeze(0)
            y_weights_tile = y_weights.unsqueeze(1) * y_weights.unsqueeze(0)
            xy_weights_tile = x_weights.unsqueeze(1) * y_weights.unsqueeze(0)

            return (
                ((x_kernel * x_weights_tile).sum() / x_weights_tile.sum())
                + ((y_kernel * y_weights_tile).sum() / y_weights_tile.sum())
                - (2 * (xy_kernel * xy_weights_tile).sum() / xy_weights_tile.sum())
            )

    @staticmethod
    def compute_kernel(x, y, gamma=None):
        """
        Gaussian RBF kernel for use in an MMD
        """
        dim = x.size(1)
        assert dim == y.size(1)
        if gamma is None:
            gamma = dim

        kernel_input = (
            (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        )  # sum over features
        return torch.exp(-gamma * kernel_input)  # (x_size, y_size)

    def compute_mmd_group(self, x, group, sample_weight=None):
        """
        Compute the MMD between data for each group
        """
        unique_groups = group.unique()
        mmd = torch.tensor([0.0], dtype=torch.float).to(self.device)
        if len(unique_groups) == 1:
            return mmd
        i = 0
        if self.config_dict["group_regularization_mode"] == "overall":
            for the_group in unique_groups:
                mmd = mmd + self.compute_mmd(
                    x[group == the_group],
                    x,
                    x_weights=sample_weight[group == the_group]
                    if sample_weight is not None
                    else None,
                    y_weights=sample_weight if sample_weight is not None else None,
                )
                i = i + 1
        elif self.config_dict["group_regularization_mode"] == "group":
            for comb in itertools.combinations(unique_groups, 2):
                mmd = mmd + self.compute_mmd(
                    x[group == comb[0]],
                    x[group == comb[1]],
                    x_weights=sample_weight[group == comb[0]]
                    if sample_weight is not None
                    else None,
                    y_weights=sample_weight[group == comb[1]]
                    if sample_weight is not None
                    else None,
                )
                i = i + 1
        return mmd / i

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Partition the data on the labels and compute the group MMD
        """
        mmd = torch.FloatTensor([0.0]).to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1].unsqueeze(1)
        if self.config_dict["mmd_mode"] == "unconditional":
            mmd = mmd + self.compute_mmd_group(
                outputs,
                group,
                sample_weight=sample_weight if sample_weight is not None else None,
            )
        else:
            if self.config_dict["mmd_mode"] == "conditional":
                unique_labels = labels.unique()
            elif self.config_dict["mmd_mode"] == "conditional_pos":
                unique_labels = [1]
            elif self.config_dict["mmd_mode"] == "conditional_neg":
                unique_labels = [0]
            else:
                raise ValueError("Invalid option provided to mmd_mode")
            for the_label in unique_labels:
                if (labels == the_label).sum() > 0:
                    mmd = mmd + self.compute_mmd_group(
                        outputs[labels == the_label],
                        group[labels == the_label],
                        sample_weight=sample_weight[labels == the_label]
                        if sample_weight is not None
                        else None,
                    )
                else:
                    logging.debug("Skipping regularization due to no samples")
        return mmd


class EqualMeanPredictionModel(GroupRegularizedModel):
    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "mean_prediction_mode": "conditional"
            # "conditional" -> eq_odds,
            # "conditional_pos" -> eq_opportunity_pos,
            # "conditional_neg" -> eq_opportunity_neg,
            # "unconditional" -> demographic_parity
        }
        return {**config_dict, **update_dict}

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Partition the data on the labels and compute the difference in means
        """
        result = torch.FloatTensor([0.0]).to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1]

        if self.config_dict["mean_prediction_mode"] == "unconditional":
            result = result + self.compute_group_regularization_loss_helper(
                outputs,
                labels,
                group,
                sample_weight=sample_weight if sample_weight is not None else None,
            )
        else:
            if self.config_dict["mean_prediction_mode"] == "conditional":
                unique_labels = labels.unique()
            elif self.config_dict["mean_prediction_mode"] == "conditional_pos":
                unique_labels = [1]
            elif self.config_dict["mean_prediction_mode"] == "conditional_neg":
                unique_labels = [0]
            else:
                raise ValueError("Invalid option provided to mean_prediction_mode")
            for the_label in unique_labels:
                if (labels == the_label).sum() > 0:
                    result = result + self.compute_group_regularization_loss_helper(
                        outputs[labels == the_label],
                        labels[labels == the_label],
                        group[labels == the_label],
                        sample_weight=sample_weight[labels == the_label]
                        if sample_weight is not None
                        else None,
                    )
                else:
                    logging.debug("Skipping regularization due to no samples")

        return result

    def compute_group_regularization_loss_helper(
        self, outputs, labels, group, sample_weight=None
    ):
        result = torch.FloatTensor([0.0]).to(self.device)
        unique_groups = group.unique()
        i=1
        for i, the_group in enumerate(unique_groups):
            result = result + (
                (
                    weighted_mean(
                        outputs[group == the_group],
                        sample_weight[group == the_group]
                        if sample_weight is not None
                        else None,
                    )
                    - weighted_mean(
                        outputs, sample_weight if sample_weight is not None else None
                    )
                )
                ** 2
            )
        result = result / i
        return result


class GroupIRMModel(GroupRegularizedModel):
    """
    Applies IRMv1 over groups
    """

    def compute_penalty(self, outputs, labels):
        # https://github.com/facebookresearch/InvariantRiskMinimization/blob/master/code/colored_mnist/main.py#L107

        scale = torch.FloatTensor([1.0]).requires_grad_().to(self.device)
        loss = self.criterion(outputs * scale, labels)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad ** 2)

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        # (TODO) implemented sample_weight
        result = torch.FloatTensor([0.0]).to(self.device)
        unique_groups = group.unique()
        for i, the_group in enumerate(unique_groups):
            the_penalty = self.compute_penalty(
                outputs[group == the_group], labels[group == the_group]
            )
            result = result + the_penalty
            logging.debug(f"Group: {the_group}, Penalty: {the_penalty}")
        return result


class EqualThresholdRateModel(GroupRegularizedModel):
    """
    A model that penalizes conditional prediction parity based on a threshold(s)
    (TODO) Harmonize this interface with EqualThresholdRateModel
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "threshold_mode": "conditional",
            # "conditional" -> eq_odds,
            # "conditional_pos" -> eq_opportunity_pos, (parity in fpr/specificity)
            # "conditional_neg" -> eq_opportunity_neg, (parity in fnr/recall)
            # "unconditional" -> demographic_parity,
            "thresholds": [0.5],
        }
        return {**config_dict, **update_dict}

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        result = torch.FloatTensor([0.0]).to(self.device)
        outputs = F.log_softmax(outputs, dim=1)[:, -1]

        for threshold in self.config_dict.get("thresholds", [0.5]):
            threshold = torch.FloatTensor([threshold]).to(self.device)
            threshold = torch.log(threshold)

            if self.config_dict["threshold_mode"] == "unconditional":
                result = result + self.compute_group_regularization_loss_helper(
                    outputs,
                    labels,
                    group,
                    sample_weight=sample_weight if sample_weight is not None else None,
                    threshold=threshold,
                )
            else:
                if self.config_dict["threshold_mode"] == "conditional":
                    unique_labels = labels.unique()
                elif self.config_dict["threshold_mode"] == "conditional_pos":
                    unique_labels = [1]
                elif self.config_dict["threshold_mode"] == "conditional_neg":
                    unique_labels = [0]
                else:
                    raise ValueError("Invalid option provided to threshold_mode")
                for the_label in unique_labels:
                    if (labels == the_label).sum() > 0:
                        result = result + self.compute_group_regularization_loss_helper(
                            outputs[labels == the_label],
                            labels[labels == the_label],
                            group[labels == the_label],
                            sample_weight=sample_weight[labels == the_label]
                            if sample_weight is not None
                            else None,
                            threshold=threshold,
                        )
                    else:
                        logging.debug("Skipping regularization due to no samples")

        return result

    def compute_group_regularization_loss_helper(
        self,
        outputs,
        labels,
        group,
        sample_weight=None,
        threshold=None,
        surrogate="logistic",
    ):
        if threshold is None:
            raise ValueError("Threshold can not be None")

        result = torch.FloatTensor([0.0]).to(self.device)
        unique_groups = group.unique()
        surrogate_fn = get_surrogate(surrogate)
        i=1
        for i, the_group in enumerate(unique_groups):
            result = result + (
                (
                    weighted_mean(
                        surrogate_fn(outputs[group == the_group] - threshold),
                        sample_weight[group == the_group]
                        if sample_weight is not None
                        else None,
                    )
                    - weighted_mean(surrogate_fn(outputs - threshold), sample_weight,)
                )
                ** 2
            )
        result = result / i
        return result


class GroupMetricRegularizedModel(GroupRegularizedModel):
    """
    A model that minimizes the difference in a metric across groups.
    """

    def get_default_config(self):
        config_dict = super().get_default_config()
        update_dict = {
            "group_regularization_mode": "overall",
            "threshold": 0.5,
            "surrogate_scale": 1.0,
        }
        return {**config_dict, **update_dict}

    def compute_group_regularization_loss(
        self, outputs, labels, group, sample_weight=None
    ):
        """
        Computes the group regularization
        """
        unique_groups = group.unique()
        result = torch.FloatTensor([0.0]).to(self.device)
        num_groups = len(unique_groups)
        if (num_groups == 1) or (labels.sum() == 0):
            warnings.warn(
                "Skipping group regularization because either one group or no outcomes in batch"
            )
            return result

        if self.config_dict["group_regularization_mode"] == "overall":
            # Regularize discrepancy in the value of the metric on groups vs. whole population
            try:
                overall_metric = self.compute_metric(
                    outputs, labels, sample_weight=sample_weight
                )
            except MetricUndefinedError:
                logging.debug("Warning: metric undefined")
                return result

            for i, the_group in enumerate(unique_groups):
                # if labels[group == the_group].sum() > 0:
                try:
                    result_group = self.compute_metric(
                        outputs[group == the_group],
                        labels[group == the_group],
                        sample_weight=sample_weight[group == the_group]
                        if sample_weight is not None
                        else None,
                    )
                    result = result + ((result_group - overall_metric) ** 2)
                except MetricUndefinedError:
                    logging.debug("Warning: metric undefined")

        elif self.config_dict["group_regularization_mode"] == "group":
            # Regularize discrepancy in the value of the metric between groups pairwise
            # (TODO) Add error handling from compute_metric
            for group_0, group_1 in itertools.combinations(unique_groups, 2):
                if (labels[group == group_0].sum() > 0) and (
                    labels[group == group_1].sum() > 0
                ):
                    result = (
                        result
                        + (
                            (
                                self.compute_metric(
                                    outputs[group == group_0],
                                    labels[group == group_0],
                                    sample_weight=sample_weight[group == group_0]
                                    if sample_weight is not None
                                    else None,
                                )
                                - self.compute_metric(
                                    outputs[group == group_1],
                                    labels[group == group_1],
                                    sample_weight=sample_weight[group == group_1]
                                    if sample_weight is not None
                                    else None,
                                )
                            )
                        )
                        ** 2
                    )
        result = result / num_groups
        return result

    def compute_metric(self, outputs, labels, sample_weight=None):
        """
        Computes the value of the metric on a batch
        """
        raise NotImplementedError

    # def surrogate_fn(self, x):
    # (TODO) delete this
    # return self.sigmoid(x, surrogate_scale=self.config_dict["surrogate_scale"])


class EqualAUCModel(GroupMetricRegularizedModel):
    """
    A model that optimizes for equal AUC across groups
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return roc_auc_score_surrogate(
            outputs=outputs, labels=labels, sample_weight=sample_weight,
        )


class EqualPrecisionModel(GroupMetricRegularizedModel):
    """
    A model that optimizes for equal precision across groups
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return precision_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
        )


class EqualTPRModel(GroupMetricRegularizedModel):
    """
    A model that optimizes for equal TPR across groups
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return tpr_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
        )


class EqualFPRModel(GroupMetricRegularizedModel):
    """
    A model that optimizes for equal TPR across groups
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return fpr_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
        )


class EqualPositiveRateModel(GroupMetricRegularizedModel):
    """
    A model that optimizes for equal TPR across groups
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return positive_rate_surrogate(
            outputs=outputs,
            labels=labels,
            sample_weight=sample_weight,
            threshold=self.config_dict.get("threshold"),
        )

class EqualLossModel(GroupMetricRegularizedModel):
    """
    Model regularized to have equal mean loss across groups.
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return weighted_cross_entropy_loss(
            outputs=outputs, labels=labels, sample_weight=sample_weight
        )


class EqualBaselinedLossModel(GroupMetricRegularizedModel):
    """
    Model regularized to have equal mean loss across groups.
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        return baselined_loss(
            outputs=outputs, labels=labels, sample_weight=sample_weight
        )


class EqualBrierScoreModel(GroupMetricRegularizedModel):
    """
    Model regularized to have equal Brier score across groups.
    """

    def compute_metric(self, outputs, labels, sample_weight=None):
        outputs = F.softmax(outputs, dim=1)[:, -1]
        return self.brier_score_loss(outputs, labels, sample_weight=sample_weight)

    def brier_score_loss(self, outputs, labels, sample_weight=None):
        return weighted_mean((outputs - labels) ** 2, sample_weight=sample_weight)


## Bilevel optimization models ##
class GroupAdversarialModel(BilevelModel, FixedWidthModel):
    def __init__(self, *args, **kwargs):
        if kwargs.get("output_dim_discriminator") is None:
            kwargs["output_dim_discriminator"] = kwargs.get("num_groups")
        print(kwargs["output_dim_discriminator"])
        super().__init__(*args, **kwargs)

    def get_default_config(self):
        """
        Defines default hyperparameters that may be overwritten.
        """
        config_dict = super().get_default_config()
        update_dict = {
            "adversarial_mode": "unconditional",
            "lr_discriminator": 1e-3,
            "lambda_group_regularization": 1e-1,
            "num_hidden_discriminator": 1,
            "hidden_dim_discriminator": 32,
            "output_dim_discriminator": None,  # specify at initialization
            "drop_prob_discriminator": 0.0,
            "normalize_discriminator": False,
            "spectral_norm": True,
            "sparse": True,
            "sparse_mode": "csr",
            "print_grads": False,
        }

        return {**config_dict, **update_dict}

    def print_grads(self):
        for name, param in self.models_aux["discriminator"].named_parameters():
            if param.requires_grad and param.grad is not None:
                logging.info("{}: {}".format(name, param.grad.mean()))

    def get_loss_names(self):
        return ["loss", "supervised", "discriminator", "discriminator_alt"]

    def init_optimizers_aux(self):
        return {
            "discriminator": torch.optim.Adam(
                [{"params": self.models_aux["discriminator"].parameters()}],
                lr=self.config_dict["lr_discriminator"],
            )
        }

    def init_models_aux(self):
        models_aux = {
            "discriminator": FeedforwardNet(
                in_features=self.config_dict["output_dim"]
                + (1 * (self.config_dict["adversarial_mode"] == "conditional")),
                hidden_dim_list=self.config_dict["num_hidden_discriminator"]
                * [self.config_dict["hidden_dim_discriminator"]],
                output_dim=self.config_dict["output_dim_discriminator"],
                drop_prob=self.config_dict["drop_prob_discriminator"],
                normalize=self.config_dict["normalize_discriminator"],
                sparse=False,
                sparse_mode=None,
                resnet=self.config_dict.get("resnet", False),
                spectral_norm=self.config_dict.get("spectral_norm", False),
            )
        }
        for model in models_aux.values():
            model.apply(self.weights_init)
            model.to(self.device)
        return models_aux

    def get_transform_batch_keys(self):
        """
        Returns the names of the list of tensors that sent to device
        """
        result = super().get_transform_batch_keys()
        result = result + ["group"]

    def forward_on_batch_helper(self, the_data):
        # Run data through the model
        outputs = self.model(the_data["features"])

        # Run data through the discriminator
        if self.config_dict["adversarial_mode"] == "conditional":
            inputs_discriminator = torch.cat(
                (
                    F.log_softmax(outputs, dim=1),
                    torch.unsqueeze(the_data["labels"].to(torch.float), dim=1),
                ),
                dim=1,
            )
        else:
            inputs_discriminator = F.log_softmax(outputs, dim=1)

        outputs_discriminator = self.models_aux["discriminator"](inputs_discriminator)

        if self.config_dict.get("weighted_loss"):
            loss_dict_batch = {
                "supervised": self.criterion(
                    outputs, the_data["labels"], sample_weight=the_data["weights"]
                ),
                "discriminator": self.criterion(
                    outputs_discriminator,
                    the_data["group"],
                    sample_weight=the_data["weights"],
                ),
            }
        else:
            loss_dict_batch = {
                "supervised": self.criterion(outputs, the_data["labels"]),
                "discriminator": self.criterion(
                    outputs_discriminator, the_data["group"]
                ),
            }
        loss_dict_batch["discriminator_alt"] = torch.exp(
            -loss_dict_batch["discriminator"]
        )
        return loss_dict_batch, outputs

    def forward_on_batch(self, the_data):
        loss_dict_batch, outputs = self.forward_on_batch_helper(the_data)

        loss_dict_batch["loss"] = (
            loss_dict_batch["supervised"]
            - self.config_dict["lambda_group_regularization"]
            * loss_dict_batch["discriminator"]
        )

        return loss_dict_batch, outputs

    def update_models_aux(self, the_data):
        loss_dict_batch, _ = self.forward_on_batch_helper(the_data)
        loss_dict_batch["discriminator"].backward()
        self.optimizers_aux["discriminator"].step()

