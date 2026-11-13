"""Compatibility CLI. Prefer python -m repoquest index."""
if __name__ == "__main__":
    import sys
    from repoquest import main
    main(["index", *sys.argv[1:]])
