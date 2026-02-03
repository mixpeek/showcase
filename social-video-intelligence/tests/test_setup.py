"""Tests for setup configuration."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup.config import Config, get_config


class TestConfig:
    """Test configuration management."""

    def test_config_defaults(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {"MIXPEEK_API_KEY": "test_key"}, clear=False):
            config = Config()
            assert config.api_key == "test_key"
            assert config.api_base == "https://api.mixpeek.com"
            assert config.namespace_id is None
            assert config.bucket_id is None

    def test_config_validation_fails_without_key(self):
        """Test that validation fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove MIXPEEK_API_KEY if it exists
            if "MIXPEEK_API_KEY" in os.environ:
                del os.environ["MIXPEEK_API_KEY"]

            config = Config()
            config.api_key = ""  # Explicitly set to empty

            with pytest.raises(ValueError, match="MIXPEEK_API_KEY"):
                config.validate()

    def test_config_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        with patch.dict(os.environ, {"MIXPEEK_API_KEY": "test_key"}, clear=False):
            config = Config()
            config.namespace_id = "ns_test123"
            config.bucket_id = "bkt_test456"

            # Override config file path
            config_file = tmp_path / "config.json"
            with patch.object(config, 'config_file', config_file):
                config.save()
                assert config_file.exists()

                # Load into new config
                new_config = Config()
                with patch.object(new_config, 'config_file', config_file):
                    new_config.load()
                    assert new_config.namespace_id == "ns_test123"
                    assert new_config.bucket_id == "bkt_test456"


class TestDataFiles:
    """Test data file loading."""

    def test_sample_videos_json_valid(self):
        """Test that sample_videos.json is valid JSON."""
        import json

        data_file = Path(__file__).parent.parent / "data" / "sample_videos.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert all("url" in item for item in data)
            assert all("title" in item for item in data)

    def test_brands_json_valid(self):
        """Test that brands.json is valid JSON."""
        import json

        data_file = Path(__file__).parent.parent / "data" / "brands.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert all("id" in item for item in data)
            assert all("name" in item for item in data)

    def test_sentiments_json_valid(self):
        """Test that sentiments.json is valid JSON."""
        import json

        data_file = Path(__file__).parent.parent / "data" / "sentiments.json"
        if data_file.exists():
            with open(data_file) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
            assert all("id" in item for item in data)
            assert all("label" in item for item in data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
