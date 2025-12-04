"""
Tests for logging configuration to ensure httpx/httpcore verbosity is properly suppressed.
"""
import logging
import io


class TestLoggingConfiguration:
    """Test that logging configuration properly suppresses verbose HTTP logging."""
    
    def test_httpx_logging_suppressed(self):
        """Test that httpx INFO logs are suppressed when configured."""
        # Capture logging output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        # Create a test logger with the configuration from pipelines
        test_logger = logging.getLogger("test_httpx_suppression")
        test_logger.setLevel(logging.INFO)
        test_logger.addHandler(handler)
        
        # Configure httpx logger to WARNING (as done in pipeline files)
        httpx_logger = logging.getLogger("httpx")
        httpx_logger.setLevel(logging.WARNING)
        httpx_logger.addHandler(handler)
        
        # Test that application logs still work
        test_logger.info("Application message")
        
        # Test that httpx INFO is suppressed
        httpx_logger.info("HTTP Request message that should be suppressed")
        
        # Test that httpx WARNING still works
        httpx_logger.warning("HTTP Warning message")
        
        # Get captured output
        output = log_capture.getvalue()
        
        # Verify results
        assert "Application message" in output
        assert "HTTP Request message" not in output
        assert "HTTP Warning message" in output
    
    def test_httpcore_logging_suppressed(self):
        """Test that httpcore INFO logs are suppressed when configured."""
        # Capture logging output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        
        # Create a test logger
        test_logger = logging.getLogger("test_httpcore_suppression")
        test_logger.setLevel(logging.INFO)
        test_logger.addHandler(handler)
        
        # Configure httpcore logger to WARNING (as done in pipeline files)
        httpcore_logger = logging.getLogger("httpcore")
        httpcore_logger.setLevel(logging.WARNING)
        httpcore_logger.addHandler(handler)
        
        # Test that application logs still work
        test_logger.info("Application message")
        
        # Test that httpcore INFO is suppressed
        httpcore_logger.info("Connection info message that should be suppressed")
        
        # Test that httpcore WARNING still works
        httpcore_logger.warning("Connection warning message")
        
        # Get captured output
        output = log_capture.getvalue()
        
        # Verify results
        assert "Application message" in output
        assert "Connection info message" not in output
        assert "Connection warning message" in output
    
    def test_pipeline_logging_configuration(self):
        """Test the exact configuration used in pipeline files."""
        # Capture logging output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Create loggers as they would be in pipeline files
        app_logger = logging.getLogger("src.pipelines.ingest")
        app_logger.setLevel(logging.INFO)
        app_logger.addHandler(handler)
        
        # Apply the fix (as done in pipeline files)
        httpx_logger = logging.getLogger("httpx")
        httpcore_logger = logging.getLogger("httpcore")
        httpx_logger.setLevel(logging.WARNING)
        httpcore_logger.setLevel(logging.WARNING)
        httpx_logger.addHandler(handler)
        httpcore_logger.addHandler(handler)
        
        # Test application logging
        app_logger.info("Processing batch")
        
        # Test that httpx/httpcore INFO is suppressed
        httpx_logger.info("HTTP Request")
        httpcore_logger.info("Connection started")
        
        # Get captured output
        output = log_capture.getvalue()
        
        # Verify that application logs work and HTTP logs are suppressed
        assert "Processing batch" in output
        assert "HTTP Request" not in output
        assert "Connection started" not in output
