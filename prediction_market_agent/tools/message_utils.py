import zlib

from prediction_market_agent_tooling.tools.hexbytes_custom import HexBytes


def compress_message(message: str) -> bytes:
    """Used to reduce size of the message before sending it to reduce gas costs."""
    return zlib.compress(message.encode(), level=zlib.Z_BEST_COMPRESSION)


def decompress_message(message: bytes) -> str:
    return zlib.decompress(message).decode()


def unzip_message_else_do_nothing(data_field: str) -> str:
    """We try decompressing the message, else we return the original data field."""
    try:
        return decompress_message(HexBytes(data_field))
    except Exception:
        return data_field
