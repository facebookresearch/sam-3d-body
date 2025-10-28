# Publishing Guide for SAM 3D Body

This document provides instructions for maintainers on how to publish SAM 3D Body to PyPI.

## Prerequisites

1. **Install build tools:**
   ```bash
   pip install build twine
   ```

2. **PyPI account:**
   - Create an account on [PyPI](https://pypi.org)
   - Create an account on [TestPyPI](https://test.pypi.org) for testing
   - Generate API tokens for both (recommended over passwords)

3. **Configure credentials:**
   Create or update `~/.pypirc`:
   ```ini
   [distutils]
   index-servers =
       pypi
       testpypi

   [pypi]
   username = __token__
   password = pypi-YOUR-API-TOKEN-HERE

   [testpypi]
   username = __token__
   password = pypi-YOUR-TESTPYPI-TOKEN-HERE
   ```

## Pre-Release Checklist

Before publishing a new version:

1. **Update version number** in `sam_3d_body/__init__.py` and `pyproject.toml`
2. **Update CHANGELOG** (if you have one) with new features, fixes, and breaking changes
3. **Run tests** to ensure everything works
4. **Update documentation** if there are API changes
5. **Commit all changes** and tag the release:
   ```bash
   git add .
   git commit -m "Release v1.0.0"
   git tag v1.0.0
   git push origin main --tags
   ```

## Build the Package

Clean previous builds and create new distribution files:

```bash
# Clean previous builds
rm -rf build/ dist/ *.egg-info

# Build the package
python -m build
```

This creates:
- `dist/sam-3d-body-1.0.0.tar.gz` (source distribution)
- `dist/sam_3d_body-1.0.0-py3-none-any.whl` (wheel distribution)

## Test the Package Locally

Before uploading to PyPI, test the package locally:

```bash
# Install in a fresh virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from the wheel
pip install dist/sam_3d_body-1.0.0-py3-none-any.whl

# Test imports
python -c "from sam_3d_body import build_sam_3d_body_model, SAM3DBodyEstimator; import sam_3d_body; print(sam_3d_body.__version__)"

# Deactivate and clean up
deactivate
rm -rf test_env
```

## Upload to TestPyPI (Recommended)

Test the upload process with TestPyPI first:

```bash
python -m twine upload --repository testpypi dist/*
```

Then test installation from TestPyPI:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sam-3d-body
```

Note: The `--extra-index-url` is needed because TestPyPI doesn't have all dependencies.

## Upload to PyPI

Once everything is tested and working:

```bash
python -m twine upload dist/*
```

Verify the upload at: https://pypi.org/project/sam-3d-body/

## Post-Release

1. **Test installation from PyPI:**
   ```bash
   pip install sam-3d-body
   pip install sam-3d-body[vis]
   pip install sam-3d-body[full]
   ```

2. **Update documentation** if needed

3. **Announce the release** on relevant channels

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions
- **PATCH** version for backwards-compatible bug fixes

Examples:
- `1.0.0` - Initial stable release
- `1.1.0` - Added new features, backwards compatible
- `1.1.1` - Bug fixes
- `2.0.0` - Breaking API changes

## Troubleshooting

### Build Errors

If you get errors during build:
```bash
# Update build tools
pip install --upgrade build setuptools wheel

# Check pyproject.toml for syntax errors
python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

### Upload Errors

- **403 Forbidden:** Check your API token and permissions
- **400 Bad Request:** Version might already exist on PyPI (you cannot overwrite)
- **File already exists:** Delete the version from PyPI or increment version number

### Import Errors After Installation

- Verify package structure: `tar -tzf dist/sam-3d-body-1.0.0.tar.gz`
- Check that `sam_3d_body/__init__.py` exports are correct
- Ensure all Python files are included in the package

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
