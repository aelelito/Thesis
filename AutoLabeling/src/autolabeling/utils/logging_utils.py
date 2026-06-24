"""
Output suppression for noisy third-party model libraries.

SAM3, SAM3D Body and SAM3D Objects all print tqdm bars and INFO logs to
stderr during inference. This context manager suppresses that output while
letting exceptions propagate normally.

Usage
-----
    from .utils.logging_utils import suppress_output

    with suppress_output():
        result = model.run(...)
"""
import contextlib
import logging
import os
import warnings


@contextlib.contextmanager
def suppress_output():
    """
    Suppress all output during model inference — Python and C-level.

    - C-level stdout/stderr (Warp, spconv banners): redirected to /dev/null via
      os.dup2() at the file-descriptor level, which catches writes that bypass
      Python's sys.stdout/sys.stderr entirely
    - tqdm / Python print: also silenced by the fd-level redirect
    - Python logging (INFO and below): logging.disable() cuts in before any
      handler is invoked, even handlers that captured sys.stderr at startup
    - loguru (SAM3D Objects uses "| INFO |" format): logger.disable("") /
      logger.enable("") — loguru ignores logging.disable()
    - Python warnings (FutureWarning, UserWarning from timm/torch):
      warnings.catch_warnings() + simplefilter('ignore')
    - Exceptions are never swallowed — fds are restored in the finally block
    """
    # ── Python-level suppression ──────────────────────────────────────────────
    logging.disable(logging.INFO)

    _loguru = None
    try:
        from loguru import logger as _loguru
        _loguru.disable('')
    except ImportError:
        pass

    # ── C-level fd suppression ────────────────────────────────────────────────
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        logging.disable(logging.NOTSET)
        if _loguru is not None:
            _loguru.enable('')
