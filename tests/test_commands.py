import sys
import unittest
from unittest.mock import patch


class TestCLICommands(unittest.TestCase):
    def test_cli_import(self):
        import lg_adk.cli.commands

    @patch("sys.argv", new=["lgadk", "--help"])
    def test_cli_help(self):
        import lg_adk.cli.commands as commands

        # If there is a main or CLI entrypoint, call it
        if hasattr(commands, "main"):
            try:
                commands.main()
            except SystemExit:
                pass


if __name__ == "__main__":
    unittest.main()
