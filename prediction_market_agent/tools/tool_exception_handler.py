from functools import wraps


def tool_exception_handler(map_exception_to_output: dict[Exception, str]):
    """
    Example usage:

    ```
    web_scrape_structured_handled = tool_exception_handler(map_exception_to_output={
        requests.exceptions.HTTPError: "Couldn't reach the URL."
    })(web_scrape_structured)
    ```

    Now you can provide `web_scrape_structured_handled` to the agent, and it will not crash if the URL is not reachable, but returns the pre-defined output instead.
    """

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if type(e) not in map_exception_to_output:
                    raise e
                return map_exception_to_output[type(e)]

        return wrapper

    return decorator
