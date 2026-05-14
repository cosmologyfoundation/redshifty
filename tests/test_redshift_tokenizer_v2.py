"""Tests for redshift tokenizer V2."""

import torch
import pytest
import numpy as np

from src.tokenizers.redshift_v2 import RedshiftTokenizerV2


class TestRedshiftTokenizerV2:
    def test_fitting(self):
        z = torch.linspace(0, 3.0, 1000)
        model = RedshiftTokenizerV2(n_levels=1024, d_model=32)
        model.fit(z)
        assert model.is_fitted
        assert model._embedding.weight.shape == (32, 1024)  # nn.Linear: (out, in)

    def test_encode_decode_roundtrip(self):
        z = torch.linspace(0, 3.0, 500)
        model = RedshiftTokenizerV2(n_levels=1024)
        model.fit(z)
        indices = model.encode(z)
        z_decoded = model.decode(indices)
        assert z_decoded.shape == z.shape
        max_err = (z - z_decoded).abs().max().item()
        assert max_err < 0.02, f"Max roundtrip error: {max_err}"

    def test_encode_decode_consistency(self):
        """Forward encode should match standalone encode."""
        z = torch.linspace(0, 2.0, 200)
        model = RedshiftTokenizerV2(n_levels=1024)
        model.fit(z)
        indices1 = model.encode(z)
        z_decoded1 = model.decode(indices1)
        max_err = (z - z_decoded1).abs().max().item()
        assert max_err < 0.025, f"Max roundtrip error: {max_err}"

    def test_embedding_shape(self):
        z = torch.linspace(0, 2.0, 50)
        model = RedshiftTokenizerV2(n_levels=1024, d_model=64)
        model.fit(z)
        indices = model.encode(z)
        emb = model.embed(indices)
        assert emb.shape == (50, 64)

    def test_forward_pipeline(self):
        z = torch.linspace(0, 2.0, 32)
        model = RedshiftTokenizerV2(n_levels=1024, d_model=32)
        model.fit(z)
        emb = model.forward(z, training=False)
        assert emb.shape == (32, 32)

    def test_encode_with_evidence(self):
        z = torch.tensor([0.1, 0.5, 1.0])
        model = RedshiftTokenizerV2(n_levels=256)
        model.fit(z)
        indices, one_hot = model.encode_with_evidence(z)
        assert indices.shape == (3,)
        assert one_hot.shape == (3, 256)
        assert one_hot.sum(dim=1).allclose(torch.ones(3))

    def test_rmse_improves_with_more_levels(self):
        z = torch.linspace(0, 3.0, 500)
        model_256 = RedshiftTokenizerV2(n_levels=256)
        model_256.fit(z)
        model_1024 = RedshiftTokenizerV2(n_levels=1024)
        model_1024.fit(z)

        rmse_256 = model_256.get_reconstruction_rmse(z)
        rmse_1024 = model_1024.get_reconstruction_rmse(z)
        assert rmse_1024 < rmse_256, \
            f"1024 levels ({rmse_1024:.6f}) should beat 256 ({rmse_256:.6f})"

    def test_redshift_class(self):
        z = torch.tensor([0.001, 0.005, 0.1, 0.5])
        model = RedshiftTokenizerV2(star_threshold=0.01)
        model.fit(z)
        classes = model.get_redshift_class(z)
        assert classes[0] == "star", f"{z[0]} should be star"
        assert classes[1] == "star", f"{z[1]} should be star (below threshold 0.01)"
        assert classes[2] == "galaxy", f"{z[2]} should be galaxy"
        assert classes[3] == "galaxy", f"{z[3]} should be galaxy"

    def test_bin_edges_shape(self):
        z = torch.linspace(0, 2.0, 500)
        model = RedshiftTokenizerV2(n_levels=128)
        model.fit(z)
        edges = model.get_bin_edges()
        assert edges.shape == (129,)  # n_levels + 1

    def test_different_batch_sizes(self):
        z = torch.linspace(0, 2.0, 200)
        model = RedshiftTokenizerV2(n_levels=512, d_model=16)
        model.fit(z)
        for bs in [1, 4, 16]:
            z_batch = z[:bs]
            indices = model.encode(z_batch)
            emb = model.forward(z_batch)
            assert indices.shape == (bs,)
            assert emb.shape == (bs, 16)

    def test_not_fitted_error(self):
        model = RedshiftTokenizerV2()
        z = torch.tensor([0.1, 0.5])
        with pytest.raises(RuntimeError, match="not fitted"):
            model.encode(z)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.decode(torch.tensor([0, 1]))
        with pytest.raises(RuntimeError, match="not fitted"):
            model.embed(torch.tensor([0, 1]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])