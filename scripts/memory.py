import json
import sys

from prediction_market_agent.db.db_storage import DBStorage

from functools import wraps


def my_decorator(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        print("Calling decorated function")
        output = f(*args, **kwds)
        print(f"output {output}")
        return output

    return wrapper


@my_decorator
def example():
    """Docstring"""
    print("Called example function")


def main():
    # print("oi")
    # db = DBStorage()
    # db.save("test1", {"a": "b"}, 1)
    # results = db.load("test1", sys.maxsize)
    # print(f"results {results}")
    example()


if __name__ == "__main__":
    main()
