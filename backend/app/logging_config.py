import logging
import sys
import structlog

def configure_logging():
    """
    Sets up structured logging using structlog.
    If running interactively, outputs colored, easy-to-read console logs.
    In production/headless mode, formats logs into structured JSON lines.
    """
    is_tty = sys.stdout.isatty()
    
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if is_tty:
        # Pretty console rendering for local development
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON formatting for production execution / Railway dashboard
        processors.append(structlog.processors.JSONRenderer())

    # Direct standard library logging output through structlog's pipeline
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
        force=True
    )
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
