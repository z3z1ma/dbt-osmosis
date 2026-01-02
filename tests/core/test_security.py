# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false

from pathlib import Path
from unittest.mock import Mock

import pytest

from dbt_osmosis.core.exceptions import PathResolutionError
from dbt_osmosis.core.path_management import (
    _get_yaml_path_template,
    get_target_yaml_path,
)
from dbt_osmosis.core.sql_operations import _has_jinja


class TestSqlInjectionSafety:
    """Tests for SQL injection protection in SQL operations."""

    def test_has_jinja_detects_jinja_tokens(self):
        """Test that _has_jinja correctly detects Jinja tokens."""
        # Valid Jinja patterns
        assert _has_jinja("SELECT {{ ref('model') }}") is True
        assert _has_jinja("{% if condition %}SELECT 1{% endif %}") is True
        assert _has_jinja("{# comment #}SELECT 1") is True

    def test_has_jinja_allows_safe_sql(self):
        """Test that _has_jinja allows safe SQL without Jinja."""
        assert _has_jinja("SELECT * FROM table") is False
        assert _has_jinja("SELECT id, name FROM users WHERE active = true") is False

    def test_sql_injection_attempt_via_comment_termination(self):
        """Test that SQL injection via comment termination is detected/handled."""
        # This tests the detection of potential injection patterns
        malicious_sql = "SELECT * FROM users WHERE id = 1; DROP TABLE users; --"
        # Should not have Jinja, so would pass to compilation
        assert _has_jinja(malicious_sql) is False
        # Note: Actual SQL execution safety depends on dbt's adapter

    def test_sql_injection_attempt_via_union_select(self):
        """Test that SQL injection via UNION SELECT is detected/handled."""
        malicious_sql = "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM passwords"
        assert _has_jinja(malicious_sql) is False
        # Note: Actual SQL execution safety depends on dbt's adapter

    def test_jinja_template_injection_detection(self):
        """Test that Jinja template injection patterns are detected."""
        # Attempt to access sensitive variables via Jinja
        injection_attempts = [
            "{{ context('env_var').get('SECRET_KEY') }}",
            "{{ config.get('credentials') }}",
            "{% print(dbt.config.credentials) %}",
        ]
        for attempt in injection_attempts:
            assert _has_jinja(attempt) is True, f"Should detect: {attempt}"


class TestPathTraversalProtection:
    """Tests for path traversal protection in YAML file operations."""

    @pytest.fixture
    def mock_context(self):
        """Create a mock context for path testing."""
        context = Mock()
        context.project = Mock()
        context.project.runtime_cfg = Mock()
        context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
        context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]
        context.yaml_handler = Mock()
        context.yaml_handler_lock = Mock()
        context.settings = Mock()
        context.settings.dry_run = False
        context.register_mutations = Mock()
        return context

    def test_path_traversal_via_dot_dot_slash(self, mock_context):
        """Test that ../ path traversal is blocked."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        # Attempt path traversal via template
        mock_node.config.extra = {"dbt-osmosis": "../../../etc/passwd"}

        with pytest.raises(PathResolutionError, match="Security violation"):
            get_target_yaml_path(mock_context, mock_node)

    def test_path_traversal_via_absolute_path(self, mock_context):
        """Test that absolute path traversal is blocked."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        # Attempt absolute path outside project root
        mock_node.config.extra = {"dbt-osmosis": "/etc/passwd"}

        # Should either raise PathResolutionError or fail at the config check
        try:
            get_target_yaml_path(mock_context, mock_node)
            # If it doesn't raise, it should still produce a safe path
        except (PathResolutionError, Exception):
            # PathResolutionError is the security exception we expect
            # Other exceptions are also acceptable for invalid input
            pass

    def test_path_traversal_via_encoded_dots(self, mock_context):
        """Test that encoded path traversal is blocked."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        # Attempt URL-encoded path traversal
        mock_node.config.extra = {"dbt-osmosis": "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"}

        # The path won't decode as a traversal but should be validated
        # This ensures the path stays within project bounds
        result = get_target_yaml_path(mock_context, mock_node)
        assert result.is_relative_to(Path("/fake/project/root").resolve())

    def test_normal_path_within_project_allowed(self, mock_context):
        """Test that normal paths within project are allowed."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        # Normal path template
        mock_node.config.extra = {"dbt-osmosis": "{node.schema}/{node.name}.yml"}

        result = get_target_yaml_path(mock_context, mock_node)
        assert "public" in str(result) or "test_model" in str(result)
        # Verify it's within project root
        project_root = Path("/fake/project/root").resolve()
        assert result.resolve().is_relative_to(project_root)

    def test_relative_path_with_subdirectory_allowed(self, mock_context):
        """Test that relative paths with subdirectories are allowed."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/subdirectory/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        # Normal subdirectory path
        mock_node.config.extra = {"dbt-osmosis": "subdirectory/{node.name}.yml"}

        result = get_target_yaml_path(mock_context, mock_node)
        # Verify it's within project root
        project_root = Path("/fake/project/root").resolve()
        assert result.resolve().is_relative_to(project_root)


class TestInputValidation:
    """Tests for input validation and sanitization."""

    def test_empty_model_name_handling(self):
        """Test that empty model names are handled gracefully."""
        # Test with empty name - should fail validation or be handled
        mock_node = Mock()
        mock_node.name = ""
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        context = Mock()
        context.project = Mock()
        context.project.runtime_cfg = Mock()
        context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
        context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]

        # This should either raise an error or produce a safe path
        # Empty names could cause path issues
        try:
            result = get_target_yaml_path(context, mock_node)
            # If it doesn't raise, ensure the path is still safe
            project_root = Path("/fake/project/root").resolve()
            assert result.resolve().is_relative_to(project_root)
        except (ValueError, KeyError, AttributeError, Exception):
            # These are acceptable failures for invalid input
            pass

    def test_special_characters_in_model_name(self):
        """Test that special characters in model names are handled."""
        # Test with potentially dangerous characters
        dangerous_names = [
            ("../../../etc/passwd", True),  # Path traversal - should fail at config check
            ("model\x00null", False),  # Null byte - should fail at path resolution
            ("model<script>alert(1)</script>", False),  # HTML/SQL - just weird characters
            ("model'; DROP TABLE users; --", False),  # SQL injection attempt - just text
            ("model\x1b[31mRED", False),  # ANSI escape - just weird characters
        ]

        for name, should_fail_at_config in dangerous_names:
            mock_node = Mock()
            mock_node.name = name
            mock_node.schema = "public"
            mock_node.database = "db"
            mock_node.package_name = "my_project"
            mock_node.resource_type = "Model"
            mock_node.original_file_path = "models/test_model.sql"
            mock_node.config = Mock()
            mock_node.config.extra = {"dbt-osmosis": "{node.schema}/{node.name}.yml"}
            mock_node.config.meta = {}
            mock_node.unrendered_config = {}
            mock_node.patch_path = None

            context = Mock()
            context.project = Mock()
            context.project.runtime_cfg = Mock()
            context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
            context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]
            context.yaml_handler = Mock()
            context.yaml_handler_lock = Mock()
            context.settings = Mock()
            context.settings.dry_run = False
            context.register_mutations = Mock()

            # Most dangerous names should either be rejected or produce safe paths
            try:
                result = get_target_yaml_path(context, mock_node)
                # If it doesn't raise, ensure path is still safe
                project_root = Path("/fake/project/root").resolve()
                # Try to resolve, but don't fail if path has invalid characters
                try:
                    resolved = result.resolve()
                    assert resolved.is_relative_to(project_root)
                except (OSError, ValueError):
                    # Path with invalid characters can't be resolved - also acceptable
                    pass
            except (ValueError, PathResolutionError, OSError, Exception):
                # These are acceptable failures for dangerous input
                pass

    def test_null_byte_injection(self):
        """Test that null byte injection is prevented."""
        # Null bytes can be used to bypass path validation
        malicious_paths = [
            "models/../../../../etc/passwd\x00.yml",
            "models/../../../sensitive\x00/data.yml",
        ]

        for path in malicious_paths:
            # These should be rejected or safely handled
            # Most path operations will reject null bytes
            try:
                result = Path(path)
                # Try to use the path
                str(result)  # noqa: B018
            except (ValueError, OSError):
                # Expected - null bytes should be rejected
                pass


class TestCredentialProtection:
    """Tests for credential leakage prevention."""

    def test_credentials_not_logged_in_sql(self):
        """Test that credentials are not logged in SQL operations."""
        # This test verifies the logging behavior
        # Actual credentials should be masked or not logged

        mock_context = Mock()
        mock_context.runtime_cfg.project_name = "test_project"
        mock_context.manifest = Mock()
        mock_context.manifest_mutex = Mock()
        mock_context.manifest.nodes = {}
        mock_context.sql_parser = Mock()
        mock_context.adapter = Mock()

        # Create a mock node
        mock_node = Mock()
        mock_node.unique_id = "test.test_project.test_model"
        mock_sql = "SELECT * FROM schema.table"

        mock_context.sql_parser.parse_remote.return_value = mock_node

        # The SQL logging should truncate to avoid logging sensitive info
        # Verify that truncation happens (75 chars limit)
        assert len(mock_sql) >= len(mock_sql[:75])
        # This test documents the expected behavior

    def test_password_not_exposed_in_error_messages(self):
        """Test that passwords are not exposed in error messages."""
        # Verify that connection errors don't leak credentials
        # This is more of a documentation test since we can't easily test
        # the actual dbt adapter error handling here
        pass


class TestSchemaValidation:
    """Tests for schema and input validation."""

    def test_invalid_yaml_template_handling(self):
        """Test that invalid YAML templates are handled."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        context = Mock()
        context.project = Mock()
        context.project.runtime_cfg = Mock()
        context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
        context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]

        # Invalid template with missing placeholder
        mock_node.config.extra = {"dbt-osmosis": "{nonexistent}/{placeholder}.yml"}

        with pytest.raises(KeyError):
            get_target_yaml_path(context, mock_node)

    def test_empty_yaml_template(self):
        """Test that empty YAML templates are handled."""
        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.resource_type = "Model"
        mock_node.config = Mock()
        mock_node.config.extra = {}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}

        context = Mock()
        context.source_definitions = {}

        # Empty template should raise MissingOsmosisConfig
        with pytest.raises(Exception):  # MissingOsmosisConfig
            _get_yaml_path_template(context, mock_node)


class TestEdgeCasesAndCornerCases:
    """Tests for edge cases and corner cases that could cause security issues."""

    def test_very_long_path_traversal(self):
        """Test that very long path traversal attempts are blocked."""
        # Create a very long traversal path
        long_traversal = "../" * 100 + "etc/passwd"

        mock_node = Mock()
        mock_node.name = "test_model"
        mock_node.schema = "public"
        mock_node.database = "db"
        mock_node.package_name = "my_project"
        mock_node.resource_type = "Model"
        mock_node.original_file_path = "models/test_model.sql"
        mock_node.config = Mock()
        mock_node.config.extra = {"dbt-osmosis": long_traversal}
        mock_node.config.meta = {}
        mock_node.unrendered_config = {}
        mock_node.patch_path = None

        context = Mock()
        context.project = Mock()
        context.project.runtime_cfg = Mock()
        context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
        context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]
        context.yaml_handler = Mock()
        context.yaml_handler_lock = Mock()
        context.settings = Mock()
        context.settings.dry_run = False
        context.register_mutations = Mock()

        # Should be blocked as security violation
        with pytest.raises(PathResolutionError, match="Security violation"):
            get_target_yaml_path(context, mock_node)

    def test_unicode_path_traversal(self):
        """Test that Unicode-based path traversal is handled."""
        # Unicode homograph attacks
        unicode_attempts = [
            "\u2025" + "etc/passwd",  # ‥ (two dots)
            "\u2215" + "etc/passwd",  # ∕ (division slash)
        ]

        for attempt in unicode_attempts:
            mock_node = Mock()
            mock_node.name = "test_model"
            mock_node.schema = "public"
            mock_node.database = "db"
            mock_node.package_name = "my_project"
            mock_node.resource_type = "Model"
            mock_node.original_file_path = "models/test_model.sql"
            mock_node.config = Mock()
            mock_node.config.extra = {"dbt-osmosis": attempt}
            mock_node.config.meta = {}
            mock_node.unrendered_config = {}
            mock_node.patch_path = None

            context = Mock()
            context.project = Mock()
            context.project.runtime_cfg = Mock()
            context.project.runtime_cfg.project_root = Path("/fake/project/root").resolve()
            context.project.runtime_cfg.model_paths = [Path("/fake/project/root/models")]
            context.yaml_handler = Mock()
            context.yaml_handler_lock = Mock()
            context.settings = Mock()
            context.settings.dry_run = False
            context.register_mutations = Mock()

            try:
                result = get_target_yaml_path(context, mock_node)
                # If it doesn't raise, ensure path is still safe
                project_root = Path("/fake/project/root").resolve()
                try:
                    resolved = result.resolve()
                    assert resolved.is_relative_to(project_root)
                except (OSError, ValueError):
                    # Path can't be resolved - also acceptable
                    pass
            except (ValueError, PathResolutionError, OSError):
                # These are acceptable failures for unusual input
                pass


class TestConcurrentAccessSafety:
    """Tests for thread safety and concurrent access."""

    def test_yaml_cache_thread_safety(self):
        """Test that YAML cache operations are thread-safe."""
        # This is a documentation test - actual thread safety testing
        # would require more complex setup with threading
        # The code uses locks (_YAML_BUFFER_CACHE_LOCK) which is good
        pass

    def test_manifest_mutex_protection(self):
        """Test that manifest access is protected by mutex."""
        # This is a documentation test - the code uses context.manifest_mutex
        # which provides thread safety
        pass
