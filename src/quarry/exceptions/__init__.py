class InterpreterEnvUpdateError(Exception):
    """Raised for global env of the python interpreter"""

    pass


class InterpreterDependencyError(InterpreterEnvUpdateError):
    """Raised when there's an error loading dependencies into interpreter."""

    pass


class InterpreterFunctionProcessingError(InterpreterEnvUpdateError):
    """Raised when there's an error processing function code into interpreter."""

    pass


class ToolLoadingFromDiskError(Exception):
    """Raise exception for loading tools into vectorDB from disk"""

    pass


class ToolConfigureError(Exception):
    """Raise exception when handling tools that are not in the right GeneratedTool format"""


class ImproperlyConfigured(Exception):
    """Raise for incorrect configuration."""

    pass


class DependencyError(Exception):
    """Raise for missing dependencies."""

    pass


class ConnectionError(Exception):
    """Raise for connection"""

    pass


class OTPCodeError(Exception):
    """Raise for invalid otp or not able to send it"""

    pass


class SQLRemoveError(Exception):
    """Raise when not able to remove SQL"""

    pass


class ExecutionError(Exception):
    """Raise when not able to execute Code"""

    pass


class ValidationError(Exception):
    """Raise for validations"""

    pass


class APIError(Exception):
    """Raise for API errors"""

    pass


class LLMResponseJsonFormatError(Exception):
    """Raise for response that are proper JSON Format"""

    pass
