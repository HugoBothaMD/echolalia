"""
Tests for downstream fine-tuning components.
"""

import pytest
import torch
import numpy as np

from clinical_speech_ssl.models.ssl_model import create_ssl_model
from clinical_speech_ssl.downstream.classifier import (
    DownstreamClassifier,
    DownstreamModel,
)
from clinical_speech_ssl.downstream.regressor import (
    DownstreamRegressor,
    DownstreamRegressorModel,
)
from clinical_speech_ssl.downstream.multi_task import (
    TaskConfig,
    MultiTaskHead,
    MultiTaskModel,
)


class TestDownstreamClassifier:
    """Test classification head."""
    
    @pytest.fixture
    def sample_features(self):
        """Sample encoder output."""
        return torch.randn(4, 50, 256)  # [B, T, D]
    
    @pytest.fixture
    def sample_lengths(self):
        return torch.tensor([50, 40, 30, 20])
    
    def test_binary_classification(self, sample_features, sample_lengths):
        """Test binary classification."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=1,
            task_type="binary",
        )
        
        logits = classifier(sample_features, sample_lengths)
        
        assert logits.shape == (4, 1)
    
    def test_multiclass_classification(self, sample_features, sample_lengths):
        """Test multi-class classification."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=5,
            task_type="multiclass",
        )
        
        logits = classifier(sample_features, sample_lengths)
        
        assert logits.shape == (4, 5)
    
    def test_multilabel_classification(self, sample_features, sample_lengths):
        """Test multi-label classification."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=10,
            task_type="multilabel",
        )
        
        logits = classifier(sample_features, sample_lengths)
        
        assert logits.shape == (4, 10)
    
    def test_with_hidden_layers(self, sample_features, sample_lengths):
        """Test classifier with hidden layers."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=5,
            hidden_dims=[128, 64],
            dropout=0.1,
        )
        
        logits = classifier(sample_features, sample_lengths)
        
        assert logits.shape == (4, 5)
    
    def test_attention_pooling(self, sample_features, sample_lengths):
        """Test attention pooling."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=3,
            pooling="attention",
        )
        
        logits = classifier(sample_features, sample_lengths)
        
        assert logits.shape == (4, 3)
    
    def test_compute_loss_binary(self, sample_features, sample_lengths):
        """Test binary classification loss."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=1,
            task_type="binary",
        )
        
        logits = classifier(sample_features, sample_lengths)
        labels = torch.randint(0, 2, (4,))
        
        result = classifier.compute_loss(logits, labels)
        
        assert 'loss' in result
        assert 'accuracy' in result
        assert result['loss'].requires_grad
    
    def test_compute_loss_multiclass(self, sample_features, sample_lengths):
        """Test multi-class classification loss."""
        classifier = DownstreamClassifier(
            embed_dim=256,
            num_classes=5,
            task_type="multiclass",
        )
        
        logits = classifier(sample_features, sample_lengths)
        labels = torch.randint(0, 5, (4,))
        
        result = classifier.compute_loss(logits, labels)
        
        assert 'loss' in result
        assert result['loss'].requires_grad


class TestDownstreamRegressor:
    """Test regression head."""
    
    @pytest.fixture
    def sample_features(self):
        return torch.randn(4, 50, 256)
    
    @pytest.fixture
    def sample_lengths(self):
        return torch.tensor([50, 40, 30, 20])
    
    def test_single_output(self, sample_features, sample_lengths):
        """Test single output regression."""
        regressor = DownstreamRegressor(
            embed_dim=256,
            num_outputs=1,
        )
        
        predictions = regressor(sample_features, sample_lengths)
        
        assert predictions.shape == (4, 1)
    
    def test_multi_output(self, sample_features, sample_lengths):
        """Test multi-output regression."""
        regressor = DownstreamRegressor(
            embed_dim=256,
            num_outputs=5,
        )
        
        predictions = regressor(sample_features, sample_lengths)
        
        assert predictions.shape == (4, 5)
    
    def test_output_range(self, sample_features, sample_lengths):
        """Test output range constraint."""
        regressor = DownstreamRegressor(
            embed_dim=256,
            num_outputs=1,
            output_activation="sigmoid",
            output_range=(0, 100),
        )
        
        predictions = regressor(sample_features, sample_lengths)
        
        assert predictions.min() >= 0
        assert predictions.max() <= 100
    
    def test_compute_loss(self, sample_features, sample_lengths):
        """Test regression loss computation."""
        regressor = DownstreamRegressor(
            embed_dim=256,
            num_outputs=1,
        )
        
        predictions = regressor(sample_features, sample_lengths)
        targets = torch.randn(4, 1)
        
        result = regressor.compute_loss(predictions, targets)
        
        assert 'loss' in result
        assert 'mae' in result
        assert 'rmse' in result
        assert 'correlation' in result


class TestMultiTaskHead:
    """Test multi-task learning head."""
    
    @pytest.fixture
    def sample_features(self):
        return torch.randn(4, 50, 256)
    
    @pytest.fixture
    def task_configs(self):
        return [
            TaskConfig("has_dysarthria", "binary", 1),
            TaskConfig("diagnosis", "multiclass", 5),
            TaskConfig("severity", "regression", 1, output_range=(0, 100)),
        ]
    
    def test_multi_task_forward(self, sample_features, task_configs):
        """Test multi-task forward pass."""
        head = MultiTaskHead(
            embed_dim=256,
            tasks=task_configs,
        )
        
        outputs = head(sample_features)
        
        assert "has_dysarthria" in outputs
        assert "diagnosis" in outputs
        assert "severity" in outputs
        
        assert outputs["has_dysarthria"].shape == (4, 1)
        assert outputs["diagnosis"].shape == (4, 5)
        assert outputs["severity"].shape == (4, 1)
    
    def test_selective_tasks(self, sample_features, task_configs):
        """Test predicting only specific tasks."""
        head = MultiTaskHead(
            embed_dim=256,
            tasks=task_configs,
        )
        
        outputs = head(sample_features, task_names=["diagnosis"])
        
        assert "diagnosis" in outputs
        assert "has_dysarthria" not in outputs
    
    def test_compute_loss(self, sample_features, task_configs):
        """Test multi-task loss computation."""
        head = MultiTaskHead(
            embed_dim=256,
            tasks=task_configs,
        )
        
        predictions = head(sample_features)
        targets = {
            "has_dysarthria": torch.randint(0, 2, (4,)),
            "diagnosis": torch.randint(0, 5, (4,)),
            "severity": torch.randn(4, 1) * 50 + 50,
        }
        
        losses = head.compute_loss(predictions, targets)
        
        assert "total_loss" in losses
        assert "has_dysarthria_loss" in losses
        assert "diagnosis_loss" in losses
        assert "severity_loss" in losses
    
    def test_compute_metrics(self, sample_features, task_configs):
        """Test multi-task metrics computation."""
        head = MultiTaskHead(
            embed_dim=256,
            tasks=task_configs,
        )
        
        predictions = head(sample_features)
        targets = {
            "has_dysarthria": torch.randint(0, 2, (4,)),
            "diagnosis": torch.randint(0, 5, (4,)),
            "severity": torch.randn(4, 1) * 50 + 50,
        }
        
        metrics = head.compute_metrics(predictions, targets)
        
        assert "has_dysarthria_accuracy" in metrics
        assert "diagnosis_accuracy" in metrics
        assert "severity_mae" in metrics


class TestDownstreamModel:
    """Test complete downstream models."""
    
    @pytest.fixture
    def encoder(self):
        return create_ssl_model("tiny")
    
    @pytest.fixture
    def sample_waveforms(self):
        return torch.randn(2, 16000)
    
    @pytest.fixture
    def sample_lengths(self):
        return torch.tensor([16000, 8000])
    
    def test_classification_model(self, encoder, sample_waveforms, sample_lengths):
        """Test full classification model."""
        classifier = DownstreamClassifier(
            embed_dim=encoder.config.embed_dim,
            num_classes=3,
            task_type="multiclass",
        )
        
        model = DownstreamModel(
            encoder=encoder,
            classifier=classifier,
            freeze_encoder=True,
        )
        
        logits = model(sample_waveforms, sample_lengths)
        
        assert logits.shape == (2, 3)
    
    def test_frozen_encoder(self, encoder, sample_waveforms, sample_lengths):
        """Test that encoder is properly frozen."""
        classifier = DownstreamClassifier(
            embed_dim=encoder.config.embed_dim,
            num_classes=3,
        )
        
        model = DownstreamModel(
            encoder=encoder,
            classifier=classifier,
            freeze_encoder=True,
        )
        
        # Check encoder params are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # Check classifier params are not frozen
        for param in model.classifier.parameters():
            assert param.requires_grad
    
    def test_compute_loss(self, encoder, sample_waveforms, sample_lengths):
        """Test loss computation."""
        classifier = DownstreamClassifier(
            embed_dim=encoder.config.embed_dim,
            num_classes=3,
            task_type="multiclass",
        )
        
        model = DownstreamModel(
            encoder=encoder,
            classifier=classifier,
        )
        
        labels = torch.randint(0, 3, (2,))
        result = model.compute_loss(sample_waveforms, labels, sample_lengths)
        
        assert 'loss' in result
        assert result['loss'].requires_grad


class TestMultiTaskModel:
    """Test multi-task model."""
    
    @pytest.fixture
    def encoder(self):
        return create_ssl_model("tiny")
    
    @pytest.fixture
    def task_configs(self):
        return [
            TaskConfig("binary_task", "binary", 1),
            TaskConfig("regression_task", "regression", 2),
        ]
    
    def test_forward(self, encoder, task_configs):
        """Test multi-task model forward."""
        head = MultiTaskHead(
            embed_dim=encoder.config.embed_dim,
            tasks=task_configs,
        )
        
        model = MultiTaskModel(encoder=encoder, multi_task_head=head)
        
        waveforms = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 8000])
        
        outputs = model(waveforms, lengths)
        
        assert "binary_task" in outputs
        assert "regression_task" in outputs
    
    def test_compute_loss(self, encoder, task_configs):
        """Test multi-task loss computation."""
        head = MultiTaskHead(
            embed_dim=encoder.config.embed_dim,
            tasks=task_configs,
        )
        
        model = MultiTaskModel(encoder=encoder, multi_task_head=head)
        
        waveforms = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 8000])
        targets = {
            "binary_task": torch.randint(0, 2, (2,)),
            "regression_task": torch.randn(2, 2),
        }
        
        losses = model.compute_loss(waveforms, targets, lengths)
        
        assert "total_loss" in losses


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
