import argparse
import contextlib


_ARGS_STACK = None


@contextlib.contextmanager
def args_scope(namespace=None, **kwargs):
    global _ARGS_STACK

    try:
        old_args = _ARGS_STACK
        new_args = argparse.Namespace(**kwargs)

        if namespace is not None:
            for name, value in namespace._get_kwargs():
                setattr(new_args, name, value)

        _ARGS_STACK = new_args
        yield new_args
    finally:
        _ARGS_STACK = old_args


def get_args():
    global _ARGS_STACK

    return _ARGS_STACK
