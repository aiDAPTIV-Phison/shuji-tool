import inspect


def get_call_site_info():
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    code = caller_frame.f_code
    module_name = caller_frame.f_globals["__name__"]
    class_name = None

    if "self" in caller_frame.f_locals:
        class_name = caller_frame.f_locals["self"].__class__.__name__

    function_name = code.co_name
    line_number = caller_frame.f_lineno

    if class_name:
        return f"{module_name}.{class_name}.{function_name} (line {line_number})"
    return f"{module_name}.{function_name} (line {line_number})"
