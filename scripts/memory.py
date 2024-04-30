import json
import sys

from prediction_market_agent.db.db_storage import DBStorage


def main():
    print("oi")
    db = DBStorage()
    db.save("test1", {"a": "b"}, 1)
    results = db.load("test1", sys.maxsize)
    print(f"results {results}")


if __name__ == "__main__":
    main()
