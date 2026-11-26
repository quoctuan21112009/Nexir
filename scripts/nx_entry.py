"""Entry point used by PyInstaller to produce the Windows CLI executable."""

from nexir.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
