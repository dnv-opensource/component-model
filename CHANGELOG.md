# Changelog

All notable changes to the [component-model] project will be documented in this file.<br>
The changelog format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
