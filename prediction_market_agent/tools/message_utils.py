import zlib


def compress_message(message: str) -> bytes:
    """Used to reduce size of the message before sending it to reduce gas costs."""
    return zlib.compress(message.encode(), level=zlib.Z_BEST_COMPRESSION)


def decompress_message(message: bytes) -> str:
    return zlib.decompress(message).decode()
