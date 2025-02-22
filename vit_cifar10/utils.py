import functools
import torch.nn as nn

# Global debug flag (can be changed dynamically)
DEBUG_MODE = False  

def debug_hook(fn):
    """Decorator to register a forward hook for printing output shapes when DEBUG_MODE is True."""
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_debug_hook_registered"):
            def hook(module, input, output):
                if DEBUG_MODE:  # Only print if debugging is enabled
                    print(f"{module.__class__.__name__}: Output Shape: {output.shape}")

            self.register_forward_hook(hook)
            self._debug_hook_registered = True  # Prevent duplicate hooks

        return fn(self, *args, **kwargs)

    return wrapper