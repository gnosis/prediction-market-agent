import os
from tempfile import TemporaryDirectory

from pinatapy import PinataPy
from prediction_market_agent_tooling.gtypes import IPFSCIDVersion0
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenThumbnailMapping,
)
from prediction_market_agent_tooling.tools.image_gen.market_thumbnail_gen import (
    generate_image_for_market,
)

from prediction_market_agent.utils import APIKeys


def generate_and_set_image_for_market(
    market: OmenAgentMarket, api_keys: APIKeys
) -> IPFSCIDVersion0 | None:
    image_mapping_contract = OmenThumbnailMapping()
    pinata = PinataPy(
        api_keys.pinata_api_key.get_secret_value(),
        api_keys.pinata_api_secret.get_secret_value(),
    )

    try:
        generated_image = generate_image_for_market(question=market.question)
    except Exception as e:
        if "content_policy_violation" in str(e):
            logger.warning(
                f"Content policy violation for {market.url}: {market.question}"
            )
            return None
        raise

    with TemporaryDirectory() as dir:
        image_path = os.path.join(dir, "image.png")
        generated_image.save(image_path)
        ipfs_hash = IPFSCIDVersion0(
            pinata.pin_file_to_ipfs(image_path, save_absolute_paths=False)["IpfsHash"]
        )

    image_mapping_contract.set(
        api_keys, market.market_maker_contract_address_checksummed, ipfs_hash
    )

    return ipfs_hash
