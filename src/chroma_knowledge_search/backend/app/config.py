import os
from pathlib import Path


def load_config():
    """Load configuration from TOML file or environment variables."""
    config_file = Path(".streamlit/secrets.toml")

    if config_file.exists():
        # Load from TOML file (local development)
        import tomllib

        with open(config_file, "rb") as f:
            config = tomllib.load(f)

        # Set environment variables from TOML
        os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
        os.environ["OPENAI_EMBED_MODEL"] = config["openai"]["embed_model"]
        os.environ["OPENAI_CHAT_MODEL"] = config["openai"]["chat_model"]
        os.environ["OPENAI_MODERATION_MODEL"] = config["openai"][
            "moderation_model"
        ]
        os.environ["CHROMA_API_KEY"] = config["chromadb"]["chroma_api_key"]
        os.environ["CHROMA_TENANT"] = config["chromadb"]["chroma_tenant"]
        os.environ["CHROMA_DATABASE"] = config["chromadb"]["chroma_database"]
        os.environ["CHROMA_COLLECTION"] = config["chromadb"][
            "chroma_collection"
        ]
        os.environ["DB_URL"] = config["sqlite"]["db_url"]
        os.environ["API_KEY"] = config["fastapi"]["api_key"]
        os.environ["API_BASE"] = config["fastapi"]["api_base"]
        os.environ["HOSTNAME"] = config["hosts"]["hostname"]
        os.environ["ALLOW_ORIGINS"] = ",".join(config["cors"]["allow_origins"])
    else:
        # Load from environment variables (production/Render)
        from dotenv import load_dotenv

        load_dotenv()


def get_allow_origins():
    """Get CORS allow origins as list."""
    return os.getenv("ALLOW_ORIGINS", "").split(",")
