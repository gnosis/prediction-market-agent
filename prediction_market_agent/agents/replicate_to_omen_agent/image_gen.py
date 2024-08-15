import os
from tempfile import TemporaryDirectory

from pinatapy import PinataPy
from prediction_market_agent_tooling.gtypes import ChecksumAddress, IPFSCIDVersion0
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenFixedProductMarketMakerContract,
    OmenThumbnailMapping,
)
from prediction_market_agent_tooling.tools.image_gen.market_thumbnail_gen import (
    generate_image_for_market_observed,
)
from prediction_market_agent_tooling.tools.langfuse_ import observe

from prediction_market_agent.utils import APIKeys


@observe()
def generate_and_set_image_for_market(
    market_address: ChecksumAddress,
    market_question: str,
    api_keys: APIKeys,
) -> IPFSCIDVersion0 | None:
    market_contract = OmenFixedProductMarketMakerContract(address=market_address)
    # Test that the market actually exists.
    # TODO: How can we get the question from the market contract (without using the graph)? (to check that `market_question` is correct)
    try:
        market_contract.totalSupply()
    except Exception as e:
        raise ValueError(
            f"Unable to verify contract exists for market address {market_address}."
        ) from e

    image_mapping_contract = OmenThumbnailMapping()
    pinata = PinataPy(
        api_keys.pinata_api_key.get_secret_value(),
        api_keys.pinata_api_secret.get_secret_value(),
    )

    try:
        generated_image = generate_image_for_market_observed(question=market_question)
    except Exception as e:
        # Roughly one every 30 markets triggers OpenAI's content policy violation, because of the prompt content. We can just skip those.
        if "content_policy_violation" in str(e):
            logger.warning(
                f"Content policy violation for {market_address=}: {market_question}"
            )
            return None
        raise

    with TemporaryDirectory() as dir:
        image_path = os.path.join(dir, "image.png")
        generated_image.save(image_path)
        ipfs_hash = IPFSCIDVersion0(
            pinata.pin_file_to_ipfs(image_path, save_absolute_paths=False)["IpfsHash"]
        )

    image_mapping_contract.set(api_keys, market_address, ipfs_hash)

    return ipfs_hash
