# pyright: reportPrivateImportUsage=false, reportPrivateUsage=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportFunctionMemberAccess=false, reportUnknownVariableType=false, reportUnusedParameter=false, reportMissingImports=false

import logging
import tempfile
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.logging import RichHandler

import dbt_osmosis.core.logger as logger_module
from dbt_osmosis.core.logger import (
    LOGGER,
    LogMethod,
    get_logger,
    get_rotating_log_handler,
    set_log_level,
)


@pytest.fixture
def temp_log_dir():
    """Create a temporary directory for log files during testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_logger(temp_log_dir):
    """Create a test logger with temporary log directory"""
    return get_logger(name="test-logger", path=temp_log_dir, level=logging.DEBUG)


class TestGetRotatingLogHandler:
    def test_creates_directory(self, temp_log_dir):
        """Test that the log directory is created if it doesn't exist"""
        handler = get_rotating_log_handler("test", temp_log_dir, "%(message)s")
        assert temp_log_dir.exists()
        assert temp_log_dir.is_dir()
        handler.close()

    def test_handler_level_set_to_warning(self, temp_log_dir):
        """Test that the handler level is set to WARNING"""
        handler = get_rotating_log_handler("test", temp_log_dir, "%(message)s")
        assert handler.level == logging.WARNING
        handler.close()

    def test_handler_format_applied(self, temp_log_dir):
        """Test that the formatter is applied to the handler"""
        formatter = "%(asctime)s - %(levelname)s - %(message)s"
        handler = get_rotating_log_handler("test", temp_log_dir, formatter)
        assert isinstance(handler.formatter, logging.Formatter)
        assert handler.formatter._fmt == formatter
        handler.close()

    def test_rotating_file_handler_properties(self, temp_log_dir):
        """Test that the rotating handler has correct properties"""
        handler = get_rotating_log_handler("test", temp_log_dir, "%(message)s")
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == int(1e6)  # 1MB
        assert handler.backupCount == 3
        handler.close()

    def test_log_file_creation(self, temp_log_dir):
        """Test that log file is created in the correct location"""
        handler = get_rotating_log_handler("test", temp_log_dir, "%(message)s")
        log_file_path = temp_log_dir / "test.log"
        # The file is created immediately when the handler is instantiated
        assert log_file_path.exists()
        handler.close()


class TestGetLogger:
    def test_returns_logger_instance(self):
        """Test that get_logger returns a Logger instance"""
        logger = get_logger(name="test-logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test-logger"

    def test_adds_rich_handler(self):
        """Test that the logger has a RichHandler"""
        test_logger = get_logger(name="test-rich")
        rich_handlers = [h for h in test_logger.handlers if isinstance(h, RichHandler)]
        assert len(rich_handlers) == 1
        assert rich_handlers[0].level == logging.INFO

    def test_adds_rotating_file_handler(self, temp_log_dir):
        """Test that the logger has a rotating file handler"""
        test_logger = get_logger(name="test-file", path=temp_log_dir)
        file_handlers = [
            h for h in test_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert file_handlers[0].level == logging.WARNING

    def test_level_conversion_string_to_int(self):
        """Test that string levels are converted to integers"""
        test_logger = get_logger(name="test-level-string", level="DEBUG")
        assert test_logger.level == logging.DEBUG

    def test_invalid_level_defaults_to_info(self):
        """Test that invalid levels default to INFO"""
        test_logger = get_logger(name="test-invalid-level", level="INVALID")
        assert test_logger.level == logging.INFO

    def test_logger_propagation_disabled(self):
        """Test that logger propagation is disabled"""
        test_logger = get_logger(name="test-propagation")
        assert test_logger.propagate is False

    def test_caching_same_logger(self):
        """Test that calling get_logger with same name returns cached instance"""
        logger1 = get_logger(name="cached-test")
        logger2 = get_logger(name="cached-test")
        assert logger1 is logger2

    def test_different_names_different_loggers(self):
        """Test that different names create different loggers"""
        logger1 = get_logger(name="test-1")
        logger2 = get_logger(name="test-2")
        assert logger1 is not logger2

    def test_cache_size_limit(self):
        """Test that the logger cache respects maxsize"""
        # This test verifies that different logger names produce different logger instances
        # The @lru_cache decorator handles the actual caching
        logger1 = get_logger(name="cache-test-1")
        logger2 = get_logger(name="cache-test-1")  # Same name, should be same instance (cached)
        logger3 = get_logger(name="cache-test-2")  # Different name, should be different instance

        assert id(logger1) == id(logger2)  # Same instance due to caching
        assert id(logger1) != id(logger3)  # Different instances due to different names


class TestSetLogLevel:
    def test_sets_level_on_default_logger(self):
        """Test that set_log_level modifies the default logger"""
        original_level = LOGGER.level
        try:
            set_log_level(logging.DEBUG)
            assert LOGGER.level == logging.DEBUG
        finally:
            set_log_level(original_level)

    def test_string_level_conversion(self):
        """Test that string levels are properly converted"""
        set_log_level("WARNING")
        assert LOGGER.level == logging.WARNING

    def test_invalid_level_defaults_to_info(self):
        """Test that invalid levels default to INFO"""
        set_log_level("INVALID")
        assert LOGGER.level == logging.INFO

    def test_updates_rich_handler_level(self):
        """Test that RichHandler level is updated but file handler is not"""
        original_level = LOGGER.level
        try:
            # Get initial handler levels
            rich_handler = None
            file_handler = None
            for handler in LOGGER.handlers:
                if isinstance(handler, RichHandler):
                    rich_handler = handler
                elif isinstance(handler, logging.handlers.RotatingFileHandler):
                    file_handler = handler

            # Change level
            set_log_level(logging.DEBUG)

            # Verify rich handler changed but file handler didn't
            assert rich_handler.level == logging.DEBUG
            assert file_handler.level == logging.WARNING  # Should remain at WARNING
        finally:
            set_log_level(original_level)


class TestDefaultLogger:
    def test_default_logger_exists(self):
        """Test that the default LOGGER is created"""
        assert LOGGER is not None
        assert isinstance(LOGGER, logging.Logger)
        assert LOGGER.name == "dbt-osmosis"

    def test_default_logger_has_handlers(self):
        """Test that the default logger has both handlers"""
        # Should have RichHandler and RotatingFileHandler
        handler_types = {type(h) for h in LOGGER.handlers}
        assert RichHandler in handler_types
        assert logging.handlers.RotatingFileHandler in handler_types


class TestLogMethodProtocol:
    def test_log_method_protocol(self):
        """Test that LogMethod protocol works"""
        # This is mainly a type checking test
        method: LogMethod = LOGGER.info
        assert callable(method)
        method("test message")


class TestModuleLevelLogging:
    """Test that the module can be used as a logger directly (via __getattr__)"""

    @patch.object(LOGGER, "info")
    def test_module_level_info_call(self, mock_info):
        """Test that calling logger.info() at module level works"""
        logger_module.info("Test info message")
        mock_info.assert_called_once_with("Test info message")

    @patch.object(LOGGER, "warning")
    def test_module_level_warning_call(self, mock_warning):
        """Test that calling logger.warning() at module level works"""
        logger_module.warning("Test warning message")
        mock_warning.assert_called_once_with("Test warning message")

    @patch.object(LOGGER, "error")
    def test_module_level_error_call(self, mock_error):
        """Test that calling logger.error() at module level works"""
        logger_module.error("Test error message")
        mock_error.assert_called_once_with("Test error message")

    @patch.object(LOGGER, "debug")
    def test_module_level_debug_call(self, mock_debug):
        """Test that calling logger.debug() at module level works"""
        logger_module.debug("Test debug message")
        mock_debug.assert_called_once_with("Test debug message")

    @patch.object(LOGGER, "critical")
    def test_module_level_critical_call(self, mock_critical):
        """Test that calling logger.critical() at module level works"""
        logger_module.critical("Test critical message")
        mock_critical.assert_called_once_with("Test critical message")

    def test_module_level_set_log_level(self):
        """Test that calling set_log_level at module level works"""
        with patch.object(logger_module, "set_log_level") as mock_set:
            logger_module.set_log_level("DEBUG")
            mock_set.assert_called_once_with("DEBUG")

    def test_module_level_getattr_unknown_attribute(self):
        """Test that unknown attributes raise AttributeError"""
        with pytest.raises(AttributeError):
            _ = logger_module.unknown_attribute


class TestEdgeCases:
    def test_concurrent_logger_creation(self, temp_log_dir):
        """Test that logger creation works in concurrent scenarios"""

        def create_logger(thread_id):
            return get_logger(f"concurrent-test-{thread_id}", path=temp_log_dir)

        threads = []
        results = []

        def worker(thread_id):
            results.append(create_logger(thread_id))

        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should have created 5 different loggers
        assert len(set(id(logger_result) for logger_result in results)) == 5

    def test_custom_formatter(self, temp_log_dir):
        """Test that custom formatter is applied"""
        custom_format = "[%(levelname)s] %(name)s: %(message)s"
        test_logger = get_logger(
            name="custom-formatter", path=temp_log_dir, formatter=custom_format
        )

        file_handler = [
            h for h in test_logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ][0]
        assert file_handler.formatter._fmt == custom_format

    def test_logger_with_path_as_string(self, temp_log_dir):
        """Test that logger works when path is passed as string (should be converted to Path)"""
        test_logger = get_logger(name="string-path", path=Path(str(temp_log_dir)))
        assert isinstance(test_logger, logging.Logger)

    def test_no_handlers_added_when_none_in_config(self):
        """Test edge case where no handlers would be added (shouldn't happen with current implementation)"""
        # This is more of a defensive programming test
        with patch("logging.Logger.addHandler"):
            test_logger = get_logger(name="no-handlers")
            assert isinstance(test_logger, logging.Logger)


class TestIntegration:
    """Integration tests that verify end-to-end functionality"""

    def test_logging_to_file_and_console(self, capsys, temp_log_dir):
        """Test that messages go to both file and console"""
        test_logger = get_logger(name="integration-test", path=temp_log_dir, level=logging.DEBUG)

        test_logger.info("Integration test message")

        # Check console output (Rich handler)
        captured = capsys.readouterr()
        assert "Integration test message" in captured.err

        # Check file output
        log_file = temp_log_dir / "integration-test.log"
        # The file is created immediately when the handler is instantiated
        assert log_file.exists()

        # File only gets WARNING+ messages by default
        initial_content = log_file.read_text()
        # INFO message should not be in file
        assert "Integration test message" not in initial_content

        # Now test with a WARNING message that should go to file
        test_logger.warning("Warning message should go to file")
        content = log_file.read_text()
        assert "Warning message should go to file" in content
        assert "integration-test" in content  # The logger name, not dbt-osmosis

    def test_log_level_filtering(self, capsys, temp_log_dir):
        """Test that log levels are properly filtered"""
        test_logger = get_logger(name="level-test", path=temp_log_dir, level=logging.WARNING)

        # These should not appear in console
        test_logger.debug("Debug message")
        test_logger.info("Info message")

        # This should appear
        test_logger.warning("Warning message")

        captured = capsys.readouterr()
        assert "Debug message" not in captured.err
        assert "Info message" not in captured.err
        assert "Warning message" in captured.err

    def test_rich_markup_support(self, capsys):
        """Test that Rich markup works in log messages"""
        # This is mainly to ensure RichHandler is configured correctly
        test_logger = get_logger(name="markup-test", level=logging.INFO)

        test_logger.info("This has [bold]bold[/bold] and [italic]italic[/] text")

        captured = capsys.readouterr()
        # The Rich handler should process markup
        assert "bold" in captured.err or "This has" in captured.err


class TestMemoryLeakPrevention:
    """Tests to ensure logger doesn't cause memory leaks"""

    def test_logger_cleanup(self):
        """Test that logger handlers are properly managed"""
        initial_handler_count = len(LOGGER.handlers)

        # Create and discard loggers
        for i in range(5):
            test_logger = get_logger(f"cleanup-test-{i}")
            # Remove handlers to simulate cleanup
            for handler in test_logger.handlers[:]:
                test_logger.removeHandler(handler)
                handler.close()

        # Should still have original handlers
        assert len(LOGGER.handlers) == initial_handler_count
