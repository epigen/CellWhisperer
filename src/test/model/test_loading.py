import unittest
from pathlib import Path
import torch
from cellwhisperer.validation import initialize_validation_functions
from cellwhisperer.jointemb.cellwhisperer_lightning import (
    TranscriptomeTextDualEncoderLightning,
)  # Replace with the correct import

from server.common.compute.cellwhisperer_wrapper import CellWhispererWrapper


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the model as part of the setup.
        cls.cellwhisperer = CellWhispererWrapper(
            "~/cellwhisperer/results/wandb_logging/JointEmbed_Training/f6fjywkb/checkpoints/last.ckpt"
        )
        cls.pl_model = cls.cellwhisperer.pl_model

        cls.validation_functions = initialize_validation_functions(
            batch_size=32,
            transcriptome_model_type=cls.pl_model.model.transcriptome_model.config.model_type,
            text_model_type=cls.pl_model.model.text_model.config.model_type,
        )
        # accelerate things by only running one validation function
        cls.validation_functions = {
            "zshot_cancer_gene_essentiality": cls.validation_functions[
                "zshot_cancer_gene_essentiality"
            ],
        }

    def test_model_parameters(self):
        # Check if the model parameters are the same as in the checkpoint
        checkpoint = torch.load(
            str(self.cellwhisperer.model_path), map_location=self.cellwhisperer.device
        )
        state_dict = checkpoint["state_dict"]
        for name, param in self.pl_model.named_parameters():
            if name in state_dict:
                self.assertTrue(
                    torch.equal(param.data, state_dict[name]),
                    f"Parameter {name} does not match.",
                )
            else:
                # Note: some model parameters are not in the state_dict, as they are frozen. Test them too
                print(f"Parameter {name} not found in state_dict")

    def test_validation_functions(self):
        # Run validation functions and check if results are above a certain threshold
        for val_fn_name, val_fn in self.validation_functions.items():
            with torch.no_grad():  # necessary, despite model being in eval mode
                val_results = val_fn(self.pl_model.model)
                val_metrics = val_results[0]
                for metric_name, metric_value in val_metrics.items():
                    print(
                        f"{val_fn_name}__{metric_name}: {metric_value}",
                    )
                    if (
                        val_fn_name == "zshot_cancer_gene_essentiality"
                        and metric_name == "mannwhitneyu_neglogp"
                    ):
                        self.assertGreaterEqual(
                            metric_value,
                            1,
                            f"{metric_name} below threshold for {val_fn_name}",
                        )


if __name__ == "__main__":
    # unittest.main()
    # Run the test as normal python to allow debugging with exception tracing
    test_model = TestModel()
    test_model.setUpClass()
    test_model.test_model_parameters()
    test_model.test_validation_functions()
