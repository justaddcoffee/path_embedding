"""Tests for CLI commands."""
import os
import tempfile
import pytest
from typer.testing import CliRunner
from path_embedding.cli import app

runner = CliRunner()
API_KEY_PATH = os.environ.get("OPENAI_API_KEY_PATH", "/Users/jtr4v/openai.key.another")


def test_train_command_help():
    """Test that train command shows help."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout.lower() or "drughangdb" in result.stdout.lower()


@pytest.mark.skipif(
    not os.path.exists(API_KEY_PATH),
    reason="API key file not found - skipping CLI integration test"
)
def test_train_command_integration():
    """Integration test for train command.

    Note: This makes real API calls and may be slow/expensive.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")

        result = runner.invoke(app, [
            "--data", "tests/data/sample_drugmechdb.yaml",
            "--output", model_path,
            "--test-size", "0.5",
            "--max-paths-per-indication", "1",
            "--api-key-path", API_KEY_PATH
        ])

        # Should succeed
        assert result.exit_code == 0

        # Model file should be created
        assert os.path.exists(model_path)
