import os

RUN_PAID_TESTS = os.environ.get("RUN_PAID_TESTS", "0") == "1"
