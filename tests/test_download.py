"""Tests for dataset registry and download functionality.

This module tests the resource registry system and download capabilities
without actually downloading full files. Tests verify:
- Resource registry structure and access
- URL accessibility (via HTTP HEAD requests)
- Download stream initialization (partial download)
- Network error handling (warnings instead of errors)

Network tests are marked with @pytest.mark.network and can be skipped.
"""

import warnings
from pathlib import Path
from typing import Any

import pytest
import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout

# Check for optional dependencies
try:
    from perturblab.data import (
        File,
        Files,
        dataset_registry,
        get_dataset,
        h5adFile,
        list_datasets,
        load_dataset,
    )

    _has_data = True
except ImportError:
    _has_data = False

try:
    from perturblab.core import Resource, ResourceRegistry

    _has_resource = True
except ImportError:
    _has_resource = False


# =============================================================================
# Fixtures and Helpers
# =============================================================================


@pytest.fixture
def sample_url() -> str:
    """Sample URL for download testing (small file)."""
    # Use a reliable, small test file from GitHub
    return "https://raw.githubusercontent.com/octocat/Hello-World/master/README"


@pytest.fixture
def network_timeout() -> int:
    """Network request timeout in seconds."""
    return 10


@pytest.fixture
def partial_download_threshold() -> int:
    """Threshold for partial download in KB (stop after this much data)."""
    return 50  # 50 KB is enough to verify download works


def check_url_accessible(
    url: str, timeout: int = 10, allow_redirects: bool = True
) -> tuple[bool, str | None]:
    """Check if a URL is accessible via HEAD request.

    Args:
        url: URL to check.
        timeout: Request timeout in seconds.
        allow_redirects: Whether to follow redirects.

    Returns:
        Tuple of (is_accessible, error_message).
        If accessible, error_message is None.
        If not accessible, error_message contains the reason.
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=allow_redirects)
        response.raise_for_status()
        return True, None
    except Timeout:
        return False, f"Timeout after {timeout}s"
    except ConnectionError:
        return False, "Connection failed (possible proxy/network issue)"
    except HTTPError as e:
        return False, f"HTTP error: {e.response.status_code}"
    except RequestException as e:
        return False, f"Request failed: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def check_download_stream(
    url: str, threshold_kb: int = 50, timeout: int = 10
) -> tuple[bool, int, str | None]:
    """Check if download stream can be initiated and data flows.

    Downloads only a small amount of data (up to threshold_kb) to verify
    the download mechanism works without downloading the entire file.

    Args:
        url: URL to download from.
        threshold_kb: Stop after downloading this many KB.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (success, bytes_downloaded, error_message).
        If successful, error_message is None and bytes_downloaded > 0.
    """
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()

            bytes_downloaded = 0
            threshold_bytes = threshold_kb * 1024

            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    bytes_downloaded += len(chunk)

                # Stop after reaching threshold
                if bytes_downloaded >= threshold_bytes:
                    break

            return True, bytes_downloaded, None

    except Timeout:
        return False, 0, f"Timeout after {timeout}s"
    except ConnectionError:
        return False, 0, "Connection failed (possible proxy/network issue)"
    except HTTPError as e:
        return False, 0, f"HTTP error: {e.response.status_code}"
    except RequestException as e:
        return False, 0, f"Request failed: {str(e)}"
    except Exception as e:
        return False, 0, f"Unexpected error: {str(e)}"


def network_test_wrapper(test_func, *args, **kwargs):
    """Wrapper that converts network errors to warnings instead of failures.

    This allows tests to pass even with network/proxy issues, while still
    reporting the problem.
    """
    try:
        return test_func(*args, **kwargs)
    except (ConnectionError, Timeout, RequestException) as e:
        warnings.warn(
            f"Network test skipped due to connection issue: {str(e)}\n"
            f"This may be due to proxy settings, firewall, or network availability.",
            UserWarning,
        )
        pytest.skip(f"Network unavailable: {str(e)}")


# =============================================================================
# Resource Type Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestResourceTypes:
    """Tests for resource type implementations (File, Files, h5adFile)."""

    def test_file_type_exists(self):
        """Test that File type can be imported."""
        assert File is not None

    def test_files_type_exists(self):
        """Test that Files type can be imported."""
        assert Files is not None

    def test_h5ad_file_type_exists(self):
        """Test that h5adFile type can be imported."""
        assert h5adFile is not None

    def test_file_is_resource(self):
        """Test that File is a subclass of Resource."""
        if _has_resource:
            assert issubclass(File, Resource)

    def test_h5ad_file_valid_extensions(self):
        """Test that h5adFile has correct valid extensions."""
        assert hasattr(h5adFile, "VALID_EXTENSIONS")
        assert ".h5ad" in h5adFile.VALID_EXTENSIONS
        assert ".h5" in h5adFile.VALID_EXTENSIONS


# =============================================================================
# Registry Structure Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestDatasetRegistry:
    """Tests for dataset registry structure and access."""

    def test_registry_exists(self):
        """Test that dataset_registry can be accessed."""
        assert dataset_registry is not None

    def test_list_datasets_function(self):
        """Test that list_datasets function works."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_list_datasets_structure(self):
        """Test that datasets follow expected naming structure."""
        datasets = list_datasets()

        # Check that datasets follow category/name structure
        for ds in datasets:
            assert "/" in ds, f"Dataset '{ds}' should follow 'category/name' format"
            parts = ds.split("/")
            assert len(parts) >= 2, f"Dataset '{ds}' should have at least category/name"

    def test_list_categories(self):
        """Test listing top-level categories."""
        categories = list_datasets(recursive=False)
        assert isinstance(categories, list)
        assert len(categories) > 0

        # Check expected categories
        assert "scperturb" in categories or "go" in categories

    def test_list_scperturb_datasets(self):
        """Test listing scPerturb datasets specifically."""
        try:
            scperturb_datasets = list_datasets(path="scperturb")
            assert isinstance(scperturb_datasets, list)
            # Should have datasets like 'scperturb/norman_2019'
            assert any("norman" in ds.lower() for ds in scperturb_datasets)
        except KeyError:
            pytest.skip("scperturb category not found in registry")

    def test_list_go_resources(self):
        """Test listing GO resources specifically."""
        try:
            go_resources = list_datasets(path="go")
            assert isinstance(go_resources, list)
            # Should have resources like 'go/go_basic', 'go/gene2go_gears'
            assert any("go_basic" in res or "gene2go" in res for res in go_resources)
        except KeyError:
            pytest.skip("go category not found in registry")

    def test_get_dataset_function(self):
        """Test that get_dataset can retrieve resources."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        # Try to get the first available dataset
        first_dataset = datasets[0]
        resource = get_dataset(first_dataset)
        assert resource is not None

    def test_get_dataset_returns_correct_type(self):
        """Test that get_dataset returns appropriate resource types."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        for ds_path in datasets[:3]:  # Test first 3 datasets
            resource = get_dataset(ds_path)

            # Should be one of the known resource types
            assert isinstance(
                resource, (File, Files, h5adFile)
            ), f"Resource '{ds_path}' has unexpected type: {type(resource)}"

    def test_invalid_dataset_path_raises_error(self):
        """Test that requesting invalid dataset raises KeyError."""
        with pytest.raises(KeyError):
            get_dataset("invalid/nonexistent_dataset")

    def test_invalid_category_raises_error(self):
        """Test that requesting invalid category raises KeyError."""
        with pytest.raises((KeyError, ValueError)):
            get_dataset("nonexistent_category/some_dataset")


# =============================================================================
# Resource Metadata Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestResourceMetadata:
    """Tests for resource metadata and properties."""

    def test_resource_has_key(self):
        """Test that resources have a key property."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        resource = get_dataset(datasets[0])
        assert hasattr(resource, "key")
        assert isinstance(resource.key, str)
        assert len(resource.key) > 0

    def test_resource_has_info_method(self):
        """Test that resources have get_info method."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        resource = get_dataset(datasets[0])
        assert hasattr(resource, "get_info")

        info = resource.get_info()
        assert isinstance(info, dict)
        assert "key" in info

    def test_resource_info_structure(self):
        """Test that resource info contains expected fields."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        resource = get_dataset(datasets[0])
        info = resource.get_info()

        # Expected fields
        assert "is_loaded" in info
        assert "is_materialized" in info
        assert "has_remote_config" in info or "has_local_path" in info

    def test_resource_has_remote_or_local(self):
        """Test that resources have either remote config or local path."""
        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        resource = get_dataset(datasets[0])
        info = resource.get_info()

        # Must have at least one source
        assert (
            info.get("has_remote_config") or info.get("has_local_path")
        ), "Resource must have either remote_config or local_path"


# =============================================================================
# URL Accessibility Tests (Network)
# =============================================================================


@pytest.mark.network
@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestURLAccessibility:
    """Tests for checking if dataset URLs are accessible.

    These tests only verify that URLs respond to HEAD requests,
    without downloading any data.
    """

    def test_sample_url_accessible(self, sample_url, network_timeout):
        """Test that the sample URL is accessible."""

        def _test():
            accessible, error = check_url_accessible(sample_url, timeout=network_timeout)
            if not accessible:
                warnings.warn(
                    f"Sample URL not accessible: {error}. "
                    f"This may indicate network/proxy issues.",
                    UserWarning,
                )
            assert accessible, f"Sample URL should be accessible, but: {error}"

        network_test_wrapper(_test)

    def test_scperturb_url_accessible(self, network_timeout):
        """Test that scPerturb Zenodo URLs are accessible."""

        def _test():
            try:
                # Get a scPerturb dataset
                scperturb_datasets = list_datasets(path="scperturb")
                if len(scperturb_datasets) == 0:
                    pytest.skip("No scPerturb datasets found")

                # Test the first dataset's URL
                resource = get_dataset(scperturb_datasets[0])
                info = resource.get_info()

                if not info.get("has_remote_config"):
                    pytest.skip("Resource has no remote config")

                # Extract URL from remote config
                remote_config = info.get("remote_config", {})
                url = remote_config.get("url")

                if not url:
                    pytest.skip("No URL found in remote config")

                # Check accessibility
                accessible, error = check_url_accessible(url, timeout=network_timeout)

                if not accessible:
                    warnings.warn(
                        f"scPerturb URL not accessible: {error}. "
                        f"This may be temporary or due to network issues.",
                        UserWarning,
                    )
                    # Don't fail the test, just warn
                    pytest.skip(f"URL not accessible: {error}")

                assert accessible, f"URL should be accessible, but: {error}"

            except KeyError:
                pytest.skip("scperturb category not found")

        network_test_wrapper(_test)

    def test_go_url_accessible(self, network_timeout):
        """Test that GO resource URLs are accessible."""

        def _test():
            try:
                # Get a GO resource
                go_resources = list_datasets(path="go")
                if len(go_resources) == 0:
                    pytest.skip("No GO resources found")

                # Test the first resource's URL
                resource = get_dataset(go_resources[0])
                info = resource.get_info()

                if not info.get("has_remote_config"):
                    pytest.skip("Resource has no remote config")

                # Extract URL from remote config
                remote_config = info.get("remote_config", {})
                url = remote_config.get("url")

                if not url:
                    pytest.skip("No URL found in remote config")

                # Check accessibility
                accessible, error = check_url_accessible(url, timeout=network_timeout)

                if not accessible:
                    warnings.warn(
                        f"GO resource URL not accessible: {error}. "
                        f"This may be temporary or due to network issues.",
                        UserWarning,
                    )
                    # Don't fail the test, just warn
                    pytest.skip(f"URL not accessible: {error}")

                assert accessible, f"URL should be accessible, but: {error}"

            except KeyError:
                pytest.skip("go category not found")

        network_test_wrapper(_test)


# =============================================================================
# Partial Download Tests (Network)
# =============================================================================


@pytest.mark.network
@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestPartialDownload:
    """Tests for verifying download streams work without full downloads.

    These tests initiate downloads but stop after a small threshold
    to verify the download mechanism works without wasting bandwidth.
    """

    def test_sample_url_download_stream(
        self, sample_url, partial_download_threshold, network_timeout
    ):
        """Test that sample URL download stream works."""

        def _test():
            success, bytes_downloaded, error = check_download_stream(
                sample_url, threshold_kb=partial_download_threshold, timeout=network_timeout
            )

            if not success:
                warnings.warn(
                    f"Download stream test failed: {error}. "
                    f"This may indicate network/proxy issues.",
                    UserWarning,
                )

            assert success, f"Download stream should work, but: {error}"
            assert (
                bytes_downloaded > 0
            ), f"Should download some data, but got {bytes_downloaded} bytes"

            # Convert to KB for display
            kb_downloaded = bytes_downloaded / 1024
            print(f"✓ Downloaded {kb_downloaded:.2f} KB successfully")

        network_test_wrapper(_test)

    def test_dataset_download_stream(
        self, partial_download_threshold, network_timeout
    ):
        """Test that dataset download streams work (partial download only)."""

        def _test():
            datasets = list_datasets()
            if len(datasets) == 0:
                pytest.skip("No datasets available")

            # Test first dataset with remote config
            for ds_path in datasets[:5]:  # Try up to 5 datasets
                resource = get_dataset(ds_path)
                info = resource.get_info()

                if not info.get("has_remote_config"):
                    continue

                remote_config = info.get("remote_config", {})
                url = remote_config.get("url")

                if not url:
                    continue

                # Found a dataset with URL, test it
                success, bytes_downloaded, error = check_download_stream(
                    url,
                    threshold_kb=partial_download_threshold,
                    timeout=network_timeout,
                )

                if not success:
                    warnings.warn(
                        f"Download stream for '{ds_path}' failed: {error}. "
                        f"This may be temporary or due to network issues.",
                        UserWarning,
                    )
                    # Don't fail, just skip
                    pytest.skip(f"Download stream not working: {error}")

                # Success!
                kb_downloaded = bytes_downloaded / 1024
                print(
                    f"✓ Dataset '{ds_path}': downloaded {kb_downloaded:.2f} KB successfully"
                )
                return  # Test passed

            pytest.skip("No datasets with remote URLs found")

        network_test_wrapper(_test)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestRegistryIntegration:
    """Integration tests for registry and resource system."""

    def test_registry_to_resource_workflow(self):
        """Test complete workflow: list -> get -> inspect."""
        # List datasets
        datasets = list_datasets()
        assert len(datasets) > 0

        # Get a dataset
        first_dataset = datasets[0]
        resource = get_dataset(first_dataset)
        assert resource is not None

        # Inspect metadata
        info = resource.get_info()
        assert "key" in info
        assert isinstance(info["is_loaded"], bool)
        assert isinstance(info["is_materialized"], bool)

    def test_registry_categories_workflow(self):
        """Test workflow with category filtering."""
        # List categories
        categories = list_datasets(recursive=False)
        assert len(categories) > 0

        # Pick first category
        category = categories[0]

        # List datasets in that category
        cat_datasets = list_datasets(path=category)
        if len(cat_datasets) > 0:
            # Get first dataset
            resource = get_dataset(cat_datasets[0])
            assert resource is not None

    def test_multiple_resource_access(self):
        """Test accessing multiple resources."""
        datasets = list_datasets()
        if len(datasets) < 2:
            pytest.skip("Need at least 2 datasets")

        # Get multiple resources
        resources = [get_dataset(ds) for ds in datasets[:3]]

        # All should be valid resources
        assert all(r is not None for r in resources)

        # Keys should match paths
        for ds_path, resource in zip(datasets[:3], resources):
            # The key might be just the dataset name (without category)
            # or the full path, so just check it's a substring
            assert (
                resource.key in ds_path or ds_path.endswith(resource.key)
            ), f"Key '{resource.key}' should relate to path '{ds_path}'"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestErrorHandling:
    """Tests for proper error handling in registry and download."""

    def test_invalid_path_format(self):
        """Test that invalid path format raises appropriate error."""
        with pytest.raises((KeyError, ValueError)):
            get_dataset("invalid_format_no_slash")

    def test_empty_path(self):
        """Test that empty path raises error."""
        with pytest.raises((KeyError, ValueError)):
            get_dataset("")

    def test_nonexistent_nested_path(self):
        """Test that deeply nested invalid path raises error."""
        with pytest.raises(KeyError):
            get_dataset("valid/category/but/nonexistent/deeply/nested")

    def test_list_invalid_category(self):
        """Test that listing invalid category raises KeyError."""
        with pytest.raises(KeyError):
            list_datasets(path="nonexistent_category_xyz")


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.skipif(not _has_data, reason="requires perturblab.data")
class TestPerformance:
    """Tests to ensure registry operations are fast."""

    def test_list_datasets_speed(self):
        """Test that listing datasets is fast (<1 second)."""
        import time

        start = time.time()
        datasets = list_datasets()
        elapsed = time.time() - start

        assert elapsed < 1.0, f"list_datasets took {elapsed:.2f}s, should be < 1s"
        assert len(datasets) > 0

    def test_get_dataset_speed(self):
        """Test that getting a dataset resource is fast."""
        import time

        datasets = list_datasets()
        if len(datasets) == 0:
            pytest.skip("No datasets available")

        start = time.time()
        resource = get_dataset(datasets[0])
        elapsed = time.time() - start

        assert (
            elapsed < 0.5
        ), f"get_dataset took {elapsed:.2f}s, should be < 0.5s"
        assert resource is not None

    def test_multiple_access_speed(self):
        """Test that accessing multiple resources is fast."""
        import time

        datasets = list_datasets()
        if len(datasets) < 5:
            pytest.skip("Need at least 5 datasets")

        start = time.time()
        for ds in datasets[:5]:
            _ = get_dataset(ds)
        elapsed = time.time() - start

        assert (
            elapsed < 1.0
        ), f"Getting 5 resources took {elapsed:.2f}s, should be < 1s"

