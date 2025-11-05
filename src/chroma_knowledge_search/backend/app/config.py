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

        # Set environment variables from TOML (only if not already set)
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = config["openai"]["api_key"]
        if not os.environ.get("OPENAI_EMBED_MODEL"):
            os.environ["OPENAI_EMBED_MODEL"] = config["openai"]["embed_model"]
        if not os.environ.get("OPENAI_CHAT_MODEL"):
            os.environ["OPENAI_CHAT_MODEL"] = config["openai"]["chat_model"]
        if not os.environ.get("OPENAI_MODERATION_MODEL"):
            os.environ["OPENAI_MODERATION_MODEL"] = config["openai"][
                "moderation_model"
            ]
        if not os.environ.get("CHROMA_API_KEY"):
            os.environ["CHROMA_API_KEY"] = config["chromadb"]["chroma_api_key"]
        if not os.environ.get("CHROMA_TENANT"):
            os.environ["CHROMA_TENANT"] = config["chromadb"]["chroma_tenant"]
        if not os.environ.get("CHROMA_DATABASE"):
            os.environ["CHROMA_DATABASE"] = config["chromadb"][
                "chroma_database"
            ]
        if not os.environ.get("CHROMA_COLLECTION"):
            os.environ["CHROMA_COLLECTION"] = config["chromadb"][
                "chroma_collection"
            ]
        if not os.environ.get("DB_URL"):
            os.environ["DB_URL"] = config["sqlite"]["db_url"]
        if not os.environ.get("API_KEY"):
            os.environ["API_KEY"] = config["fastapi"]["api_key"]
        if not os.environ.get("API_BASE"):
            os.environ["API_BASE"] = config["fastapi"]["api_base"]
        if not os.environ.get("HOSTNAME"):
            os.environ["HOSTNAME"] = config["hosts"]["hostname"]
        if not os.environ.get("ALLOW_ORIGINS"):
            os.environ["ALLOW_ORIGINS"] = ",".join(
                config["cors"]["allow_origins"]
            )
    else:
        # Load from environment variables (production/Render)
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # dotenv not available, skip loading
            pass
        except FileNotFoundError:
            # .env file may not exist in CI
            pass


def get_allow_origins():
    """Get CORS allow origins as list."""
    return os.getenv("ALLOW_ORIGINS", "").split(",")
