import irispy_lts   # or: import irispy  — whichever the install exposes
print(irispy_lts.__version__ if hasattr(irispy_lts, "__version__") else "no version attr")
print([x for x in dir(irispy_lts) if not x.startswith("_")])
help(irispy_lts)   # module-level docstring / usage