"""
Configuration management for Social Video Intelligence showcase.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class Config:
    """Configuration for the showcase."""

    # API Configuration
    api_key: str = field(default_factory=lambda: os.environ.get("MIXPEEK_API_KEY", ""))
    api_base: str = field(
        default_factory=lambda: os.environ.get("MIXPEEK_API_BASE", "https://api.mixpeek.com")
    )

    # Resource IDs (populated after setup)
    namespace_id: Optional[str] = None
    bucket_id: Optional[str] = None

    # Collection IDs
    visual_collection_id: Optional[str] = None
    audio_collection_id: Optional[str] = None
    text_collection_id: Optional[str] = None

    # Taxonomy IDs
    brand_taxonomy_id: Optional[str] = None
    sentiment_taxonomy_id: Optional[str] = None
    content_taxonomy_id: Optional[str] = None

    # Retriever IDs
    unified_retriever_id: Optional[str] = None
    brand_monitoring_retriever_id: Optional[str] = None

    # Cluster IDs
    narrative_cluster_id: Optional[str] = None

    # Alert IDs
    negative_brand_alert_id: Optional[str] = None

    # Reference data buckets/collections
    brand_bucket_id: Optional[str] = None
    brand_collection_id: Optional[str] = None
    brand_retriever_id: Optional[str] = None
    sentiment_bucket_id: Optional[str] = None
    sentiment_collection_id: Optional[str] = None
    sentiment_retriever_id: Optional[str] = None

    @property
    def config_file(self) -> Path:
        """Path to config file."""
        return Path(__file__).parent.parent / "config.json"

    def save(self) -> None:
        """Save configuration to file."""
        data = {
            "namespace_id": self.namespace_id,
            "bucket_id": self.bucket_id,
            "visual_collection_id": self.visual_collection_id,
            "audio_collection_id": self.audio_collection_id,
            "text_collection_id": self.text_collection_id,
            "brand_taxonomy_id": self.brand_taxonomy_id,
            "sentiment_taxonomy_id": self.sentiment_taxonomy_id,
            "content_taxonomy_id": self.content_taxonomy_id,
            "unified_retriever_id": self.unified_retriever_id,
            "brand_monitoring_retriever_id": self.brand_monitoring_retriever_id,
            "narrative_cluster_id": self.narrative_cluster_id,
            "negative_brand_alert_id": self.negative_brand_alert_id,
            "brand_bucket_id": self.brand_bucket_id,
            "brand_collection_id": self.brand_collection_id,
            "brand_retriever_id": self.brand_retriever_id,
            "sentiment_bucket_id": self.sentiment_bucket_id,
            "sentiment_collection_id": self.sentiment_collection_id,
            "sentiment_retriever_id": self.sentiment_retriever_id,
        }
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Configuration saved to {self.config_file}")

    def load(self) -> "Config":
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            print(f"Configuration loaded from {self.config_file}")
        return self

    def validate(self) -> bool:
        """Validate that API key is set."""
        if not self.api_key:
            raise ValueError(
                "MIXPEEK_API_KEY environment variable is required.\n"
                "Set it in your environment or in a .env file.\n"
                "Get your API key from https://mixpeek.com/dashboard"
            )
        return True


def get_config() -> Config:
    """Get configuration, loading from file if exists."""
    config = Config()
    config.load()
    config.validate()
    return config
