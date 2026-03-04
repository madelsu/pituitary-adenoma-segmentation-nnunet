from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainer_StrongAug(nnUNetTrainer):
    """
    Slightly stronger augmentation than default nnU-Net.

    This trainer is written to be compatible with nnU-Net v2 versions where
    get_training_transforms(...) receives keyword arguments (e.g., use_mask_for_
norm).
    """

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_range, use_dummy_2d, mirror_axes, patch_size = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # Widen rotation ranges a bit (best-effort, depending on what parent ret
urns)
        def widen(r, factor=1.5):
            if r is None:
                return None
            if isinstance(r, (tuple, list)) and len(r) == 2:
                return (r[0] * factor, r[1] * factor)
            return r

        if isinstance(rotation_range, (list, tuple)):
            rotation_range = [widen(r, 1.5) for r in rotation_range]

        return rotation_range, use_dummy_2d, mirror_axes, patch_size

    def get_training_transforms(self, *args, **kwargs):
        """
        Keep nnU-Net's default transform pipeline, but increase probabilities
        of some intensity augmentations in a safe way.
        """
        tr = super().get_training_transforms(*args, **kwargs)

        # Best-effort: increase p_per_sample for existing transforms if present
        for t in getattr(tr, "transforms", []):
            name = t.__class__.__name__.lower()

            if "gammatransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + 0.2)

            if "gaussiannoisetransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + 0.2)

            if "gaussianblurtransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + 0.2)

            if "biasfieldtransform" in name and hasattr(t, "p_per_sample"):
                t.p_per_sample = min(1.0, t.p_per_sample + 0.2)

        return tr
