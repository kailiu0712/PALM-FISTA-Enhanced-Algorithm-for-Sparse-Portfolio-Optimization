try:
    from .benchmark import main
except ImportError:  # pragma: no cover - script mode
    from benchmark import main


if __name__ == "__main__":
    main()
