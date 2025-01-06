import os
import sys
from functools import wraps
from io import StringIO


# Define the decorator
def conditional_capture(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check for CI environment variables
        is_ci = os.getenv("CI") or os.getenv("GITHUB_ACTIONS") or os.getenv("GITLAB_CI")

        if is_ci:
            # Capture output using StringIO when in CI environment
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            try:
                return func(*args, **kwargs)
            finally:
                sys.stdout = old_stdout
        else:
            return func(*args, **kwargs)

    return wrapper
