from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_MildAug(nnUNetTrainer):
    """
    Mild augmentation: stays close to nnU-Net defaults,
    with a small increase in rotation range + small increases
    in probabilities of a few intensity transforms.

    Goal: be safer for small structures (pituitary tumors) than StrongAug.
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_range, use_dummy_2d, mirror_axes, patch_size = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Slightly widen rotation ranges (smaller than StrongAug)
        def widen(r, factor=1.15):
            if r is None:
                return None
            if isinstance(r, (tuple, list)) and len(r) == 2:
                return (r[0] * factor, r[1] * factor)
            return r

        if isinstance(rotation_range, (list, tuple)):
            rotation_range = [widen(r, 1.15) for r in rotation_range]

        return rotation_range, use_dummy_2d, mirror_axes, patch_size

    def get_training_transforms(self, *args, **kwargs):
        """
        Keep nnU-Net's default transform pipeline,
        but gently increase p_per_sample for some intensity transforms.
        """
        tr = super().get_training_transforms(*args, **kwargs)

        # Smaller bump than StrongAug (+0.2 -> +0.05 to +0.10)
        bump = 0.10

        for t in getattr(tr, "transforms", []):
            name = t.__class__.__name__.lower()

            if "gammatransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + bump)

            if "gaussiannoisetransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + bump)

            if "gaussianblurtransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + bump)

            # Bias field can be risky; bump it less (or disable bump entirely)
            if "biasfieldtransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + 0.05)

        return tr
