import inspect


def obj_signature(obj):
    sig = inspect.signature(obj)

    return {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in sig.parameters.items()
        if v.kind is not v.VAR_POSITIONAL and v.kind is not v.VAR_KEYWORD
    }
