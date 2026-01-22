# Changelog

All notable changes to the [component-model] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

* -/-


## [0.3.2] - 2026-01-23

### Added
* Added new module unit.py, containing class `Unit`, a helper class to store and manage units and display units. One `Unit` object represents one scalar variable.
* Added new module range.py, containing class `Range`, a utility class to store and handle the variable range of a single-valued variable.
* Sphinx documentation:
  * Added docs for modules variable_naming.py, unit.py, range.py, enums.py and analytic.py
* Added Visual Studio Code settings

### Removed
* Removed module plotter.py

### Changed
* Updated code base with latest changes in python_project_template v0.2.6
* pyproject.toml:
  * Updated supported Python versions to 3.11, 3.12, 3.13, 3.14
  * Updated required Python version to ">= 3.11"
  * Renamed optional dependency group 'tests' to 'test' to make it uniform with crane-fmu (there also 'test' is used).
* ruff.toml:
  * Updated target Python version to "py311"
* .sourcery.yaml:
  *  Updated the lowest Python version the project supports to '3.11'
* GitHub workflow _test.yml:
  * Updated Python versions in test matrix to 3.11, 3.12, 3.13, 3.14
* GitHub workflow _test_future.yml:
  * Updated Python version in test_future to 3.15.0-alpha - 3.15.0
* GitHub workflow _build_and_publish_documentation.yml:
  * Changed 'uv sync --upgrade' to 'uv sync --frozen' to avoid unintentional package upgrades.
* Sphinx documentation:
  * Updated toctree
  * conf.py: Updated, and removed ruff rule exception on file level

### Dependencies
* Updated to ruff>=0.14.3  (from ruff>=0.6.3)
* Updated to pyright>=1.1.407  (from pyright>=1.1.378)
* Updated to sourcery>=1.40  (from sourcery>=1.22)
* Updated to numpy>=2.3  (from numpy>=2.0)
* Updated to scipy>=1.16  (from scipy>=1.15.1)
* Updated to matplotlib>=3.10  (from matplotlib>=3.9.1)
* Updated to plotly>=6.3  (from plotly>=6.0.1)
* Updated to pytest>=8.4  (from pytest>=8.3)
* Updated to pytest-cov>=7.0  (from pytest-cov>=5.0)
* Updated to Sphinx>=8.2  (from Sphinx>=8.0)
* Updated to sphinx-argparse-cli>=1.20  (from sphinx-argparse-cli>=1.17)
* Updated to sphinx-autodoc-typehints>=3.5  (from sphinx-autodoc-typehints>=2.2)
* Updated to furo>=2025.9  (from furo>=2024.8)
* Updated to pre-commit>=4.3  (from pre-commit>=3.8)
* Updated to mypy>=1.18  (from mypy>=1.11.1)
* Updated to checkout@v5  (from checkout@v4)
* Updated to setup-python@v6  (from setup-python@v5)
* Updated to setup-uv@v7  (from setup-uv@v2)
* Updated to upload-artifact@v5  (from upload-artifact@v4)
* Updated to download-artifact@v5  (from download-artifact@v4)


## [0.3.1] - 2025-12-17

### Changed
* Address breaking changes with PythonFMU's latest changes to setup_experiment() function.


## [0.3.0] - 2025-12-15

### Added
* Added the documentation file `component-development-process.rst`, outlining the recommended FMU development workflow, best practices, and the role of virtual derivatives.
* Added the example XML structures `BouncingBallStructure.xml` and `ForcedOscillator6D.xml` to showcase richer algorithm and ECCO configuration sections together with clarified causality definitions.

### Changed
* Refined the existing FMU XML examples with explicit algorithm sections, ECCO configuration, and clearer variable linkage descriptions.
* Updated `driving_force_fmu.py` to support vectorized amplitudes, frequencies, and frequency sweeps, while improving type annotations and initialization semantics.
* Changed the upper bound for the `height` variable in `bouncing_ball_3d.py` to inches to align with the rest of the example unit system.

### GitHub workflows
* Added Python 3.13 to the main CI matrix to ensure compatibility with the current stable release.
* Updated the future/experimental workflow to track Python 3.14 instead of 3.13 for forward-looking coverage.

## [0.2.0] - 2025.30.04

### Changed
* New structured variables feature with hierarchical variable organization using dot notation
* Support for derivative notation with `der(variable)` and `der(variable,n)` syntax
* Automatic handling of derivatives without explicit definitions in base models
* Variable naming conventions: `flat` and `structured` in `VariableNamingConvention` enum
* Example implementations in `axle.py`, `axle_fmu.py` and test cases in `test_structured_variables.py`


## [0.1.0] - 2024-11-08

### Changed
* Changed from `pip`/`tox` to `uv` as package manager
* README.rst : Completely rewrote section "Development Setup", introducing `uv` as package manager.
* Changed publishing workflow to use OpenID Connect (Trusted Publisher Management) when publishing to PyPI

### GitHub workflows
* (all workflows): Adapted to use `uv` as package manager
* _test_future.yml : updated Python version to 3.13.0-alpha - 3.13.0
* _test_future.yml : updated name of test job to 'test313'


## [0.0.1] - 2024-09-27

* Initial release

### Added

* added this

### Changed

* changed that

### Dependencies

* updated to some_package_on_pypi>=0.1.0

### Fixed

* fixed issue #12345

### Deprecated

* following features will soon be removed and have been marked as deprecated:
    * function x in module z

### Removed

* following features have been removed:
    * function y in module z


<!-- Markdown link & img dfn's -->
[unreleased]: https://github.com/dnv-innersource/component-model/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/dnv-innersource/component-model/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/dnv-innersource/component-model/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/dnv-innersource/component-model/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/dnv-innersource/component-model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/dnv-innersource/component-model/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/component-model/releases/tag/v0.0.1
[component-model]: https://github.com/dnv-innersource/component-model
