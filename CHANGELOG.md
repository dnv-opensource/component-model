# Changelog

All notable changes to the [component-model] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Dependencies
* Updated to ruff>=0.9.2  (from ruff>=0.6.3)
* Updated to pyright>=1.1.392  (from pyright>=1.1.378)
* Updated to sourcery>=1.31  (from sourcery>=1.22)
* Updated to numpy>=1.26  (from numpy>=1.26,<2.0)
* Updated to sympy>=1.13  (from sympy>=1.13.3)
* Updated to matplotlib>=3.10  (from matplotlib>=3.9.1)
* Updated to pytest-cov>=6.0  (from pytest-cov>=5.0)
* Updated to Sphinx>=8.1  (from Sphinx>=8.0)
* Updated to sphinx-argparse-cli>=1.19  (from sphinx-argparse-cli>=1.17)
* Updated to sphinx-autodoc-typehints>=3.0  (from sphinx-autodoc-typehints>=2.2)
* Updated to libcosimpy==0.0.2  (from libcosimpy>=0.0.2)
* Updated to pre-commit>=4.0  (from pre-commit>=3.8)
* Updated to mypy>=1.14  (from mypy>=1.11.1)
* Updated to setup-uv@v5  (from setup-uv@v2)

-/-


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
[unreleased]: https://github.com/dnv-innersource/component-model/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dnv-innersource/component-model/releases/tag/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/component-model/releases/tag/v0.0.1
[component-model]: https://github.com/dnv-innersource/component-model
