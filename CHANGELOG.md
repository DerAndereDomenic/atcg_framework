# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Possibility to flip decoded jpeg images
- Custom key codes

### Changed

- Default behavior of JPEG encoder and decoder: now flip images
- The caller of TCPClient::sendAndWait had to manually prepend the message size. This is now done automatically.

### Fixed

- Fixed linking of imgui in headless mode by removing unnecessary libraries in linking stage
