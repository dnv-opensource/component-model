# Changelog

All notable changes to the [component-model] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

-/-

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
[unreleased]: https://github.com/dnv-innersource/component-model/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dnv-innersource/component-model/releases/tag/v0.0.1...v0.1.0
[0.0.1]: https://github.com/dnv-innersource/component-model/releases/tag/v0.0.1
[component-model]: https://github.com/dnv-innersource/component-model
