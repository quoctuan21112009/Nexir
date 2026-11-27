import argparse
from pathlib import Path
import tempfile
import unittest

from nexir.compiler import NexirError
from nexir.ecosystem import ecosystem_command


class NexirEcosystemTests(unittest.TestCase):
    def test_init_creates_native_manifest_and_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            ecosystem_command(argparse.Namespace(action="init", path=Path(directory), name="my-lib"))
            root = Path(directory)
            self.assertTrue((root / "nx.toml").is_file())
            self.assertTrue((root / "src" / "main.nx").is_file())
            ecosystem_command(argparse.Namespace(action="validate", path=root))

    def test_validation_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside.nx"
            outside.write_text("def main():\n    pass\n", encoding="utf-8")
            (root / "nx.toml").write_text(
                '[package]\nname="safe"\nversion="1.0.0"\nentry="../outside.nx"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(NexirError, "inside the project"):
                ecosystem_command(argparse.Namespace(action="validate", path=root))
            outside.unlink(missing_ok=True)

    def test_validation_rejects_bad_dependency_constraint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.nx").write_text("def main():\n    pass\n", encoding="utf-8")
            (root / "nx.toml").write_text(
                '[package]\nname="safe"\nversion="1.0.0"\nentry="src/main.nx"\n'
                '[dependencies]\nother="latest"\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(NexirError, "version constraint"):
                ecosystem_command(argparse.Namespace(action="validate", path=root))


if __name__ == "__main__":
    unittest.main()
