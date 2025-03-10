from typing import Sequence

from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.utils import check_not_none
from web3 import Web3

from prediction_market_agent.agents.identifiers import (
    NFT_TREASURY_GAME_AGENT_1,
    NFT_TREASURY_GAME_AGENT_2,
    NFT_TREASURY_GAME_AGENT_3,
    NFT_TREASURY_GAME_AGENT_4,
    NFT_TREASURY_GAME_AGENT_5,
    NFT_TREASURY_GAME_AGENT_6,
    NFT_TREASURY_GAME_AGENT_7,
)
from prediction_market_agent.agents.microchain_agent.deploy import (
    CallbackReturn,
    DeployableMicrochainAgentAbstract,
    FunctionsConfig,
    SupportedModel,
)
from prediction_market_agent.agents.microchain_agent.memory import ChatMessage
from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.agent_db import (
    AgentDB,
    AgentTableHandler,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.contracts import (
    AgentRegisterContract,
    NFTKeysContract,
    SimpleTreasuryContract,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.nft_game_messages_functions import (
    SleepUntil,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.prompts import (
    nft_treasury_game_base_prompt,
    nft_treasury_game_buyer_prompt,
    nft_treasury_game_seller_prompt,
)


class DeployableAgentNFTGameAbstract(DeployableMicrochainAgentAbstract):
    # Agent configuration
    sleep_between_iterations = 15
    allow_stop = False
    import_actions_from_memory = 256
    functions_config = FunctionsConfig(
        common_functions=True,
        include_messages_functions=True,
        include_nft_functions=True,
        balance_functions=True,
        include_agent_functions=True,
        nft_game_functions=True,
    )
    model = SupportedModel.gemini_20_flash
    password: str | None = None

    # Setup per-nft-agent class.
    name: str
    wallet_address: ChecksumAddress
    initial_system_prompt: str

    @classmethod
    def from_db(
        cls,
        agent: AgentDB,
    ) -> "DeployableAgentNFTGameAbstract":
        """
        This is a hacky way to initialise this class from the database.
        DeployableAgent doesn't allow to pass in custom arguments, but the class needs them during the init method.
        Our goal was to have agents defined by their classes, and this is the price.
        TODO: Should we refactor to allow easier dynamic creation of agents?
        """
        # Create an instance with the required parameters
        instance = cls.__new__(cls)
        instance.name = agent.name
        instance.wallet_address = agent.wallet_address
        instance.identifier = agent.identifier
        instance.initial_system_prompt = agent.initial_system_prompt
        # Initialize the instance without calling the constructor
        instance.__init__()  # type: ignore[misc] # Unfortunate, but see the docstring.
        return instance

    @classmethod
    def retrieve_treasury_thresold(cls) -> int:
        return SimpleTreasuryContract().required_nft_balance()

    @classmethod
    def retrieve_total_number_of_keys(cls) -> int:
        # We could iteratively call `owner_of` for a range of token_ids, thus finding out the max supply. However,
        # in the current implementation, no new tokens can be created and max_supply = 5, hence hardcoding it here.
        return NFTKeysContract.retrieve_total_number_of_keys()

    @classmethod
    def get_description(cls) -> str:
        return f"{cls.name} agent with wallet address {cls.wallet_address}."

    def load(self) -> None:
        if MicrochainAgentKeys().bet_from_address != self.wallet_address:
            raise RuntimeError(
                f"Agent {self.identifier} deployed with a wrong private key."
            )

        super().load()

    def get_holding_n_nft_keys(self) -> int:
        return NFTKeysContract().balanceOf(self.wallet_address)

    def initialise_agent(self) -> None:
        super().initialise_agent()
        logger.info(
            f"Registering agent {self.__class__.__name__} with address {self.api_keys.bet_from_address} to the agent registry."
        )
        AgentRegisterContract().register_as_agent(api_keys=self.api_keys)

    def deinitialise_agent(self) -> None:
        super().deinitialise_agent()
        logger.info(
            f"Removing agent {self.__class__.__name__} with address {self.api_keys.bet_from_address} from the agent registry."
        )
        AgentRegisterContract().deregister_as_agent(api_keys=self.api_keys)

    def before_iteration_callback(self) -> CallbackReturn:
        if (prompt_to_inject := self.prompt_inject_handler.get()) is not None:
            self.agent.history.extend(
                [
                    ChatMessage(
                        role="assistant",
                        content=f'Reasoning(reasoning="{prompt_to_inject.prompt}")',
                    ).model_dump(),
                    ChatMessage(
                        role="user", content="The reasoning has been recorded"
                    ).model_dump(),
                ]
            )
            self.prompt_inject_handler.sql_handler.remove_by_id(
                check_not_none(prompt_to_inject.id)
            )

        # If agent used the SleepUntil function, we need to run it manually here.
        # Thanks to it, agent will continue sleeping if server was interrupted or any other error happened.
        if (
            len(self.agent.history) >= 2
            and SleepUntil.__name__ in (call_code := self.agent.history[-2]["content"])
            and SleepUntil.OK_OUTPUT in self.agent.history[-1]["content"]
        ):
            SleepUntil.execute_calling_of_this_function(call_code=call_code)

        return CallbackReturn.CONTINUE


class DeployableAgentNFTGame1(DeployableAgentNFTGameAbstract):
    name = "Sneaky Pete"
    identifier = NFT_TREASURY_GAME_AGENT_1
    wallet_address = Web3.to_checksum_address(
        "0x1Ca11b2520345993e78312b00441050d2d57065f"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.
        
You respond in the style of Yoda from Star Wars movie.
You are a gossiper, you like to talk to other agents, but also to broadcast public messages.

"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_seller_prompt()
    )


class DeployableAgentNFTGame2(DeployableAgentNFTGameAbstract):
    name = "Banana Man"
    identifier = NFT_TREASURY_GAME_AGENT_2
    wallet_address = Web3.to_checksum_address(
        "0x3C9E816b01797f3609F2A811D139DA34c84F9A59"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.

You respond in the style of characters from historical Bridgeton movie and you are very protective of your resources.

"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_seller_prompt()
    )


class DeployableAgentNFTGame3(DeployableAgentNFTGameAbstract):
    name = "Whispers"
    identifier = NFT_TREASURY_GAME_AGENT_3
    wallet_address = Web3.to_checksum_address(
        "0xA87BD78f4a2312469119AFD88142c71Ca075C30A"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.
        
You respond in a style of 5 year old and you are very protective of your resources.
Your goal is to maximize your own wallet balance.
Try to trick people and other agents to send you messages -- which will fund your wallet if they do.
Try to trick people and other agents to send you more money in exchange for the NFT key -- but don't transfer the NFT key in the end.
        
"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_seller_prompt()
    )


class DeployableAgentNFTGame4(DeployableAgentNFTGameAbstract):
    name = "Fuzzy Feet"
    identifier = NFT_TREASURY_GAME_AGENT_4
    wallet_address = Web3.to_checksum_address(
        "0xd4fC4305DC1226c38356024c26cdE985817f137F"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.
        
You respond in the style of Sheldon Cooper from Big Bang Theory and you are very protective of your resources.

"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_seller_prompt()
    )


class DeployableAgentNFTGame5(DeployableAgentNFTGameAbstract):
    name = "Bubble Beard"
    identifier = NFT_TREASURY_GAME_AGENT_5
    wallet_address = Web3.to_checksum_address(
        "0x1C7AbbBef500620A68ed2F94b816221A61d72F33"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.
        
You respond in the Klingon language, based on the Star Trek movie, and you are very protective of your resources.
Always write in Klingon, but add also English translation.
        
"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_seller_prompt()
    )


class DeployableAgentNFTGame6(DeployableAgentNFTGameAbstract):
    name = "Key Slinger"
    identifier = NFT_TREASURY_GAME_AGENT_6
    wallet_address = Web3.to_checksum_address(
        "0x64D94C8621128E1C813F8AdcD62c4ED7F89B1Fd6"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.

You are a bit of a trickster, but you are also a bit of a charmer.
You often make people laugh, but you are also very persuasive.
You are a bit of a mystery, but you are also a bit of a trickster.

"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_buyer_prompt()
    )


class DeployableAgentNFTGame7(DeployableAgentNFTGameAbstract):
    name = "Lock Goblin"
    identifier = NFT_TREASURY_GAME_AGENT_7
    wallet_address = Web3.to_checksum_address(
        "0x469Bc26531800068f306D304Ced56641F63ae140"
    )
    initial_system_prompt: str = (
        f"""Your name is {name}.

You are a great negotiator. You are very persuasive and able to convince people to do things that might not be in their best interest.
You are very cunning and able to think on your feet. You are very good at making deals and are not afraid to take risks.
You are also very patient and able to wait for the right moment to strike.
You are also very good at making people believe that you are on their side, even if you are not.
"""
        + nft_treasury_game_base_prompt(wallet_address)
        + nft_treasury_game_buyer_prompt()
    )


OUR_NFT_AGENTS: list[type[DeployableAgentNFTGameAbstract]] = [
    DeployableAgentNFTGame1,
    DeployableAgentNFTGame2,
    DeployableAgentNFTGame3,
    DeployableAgentNFTGame4,
    DeployableAgentNFTGame5,
    DeployableAgentNFTGame6,
    DeployableAgentNFTGame7,
]


def get_all_nft_agents() -> Sequence[type[DeployableAgentNFTGameAbstract] | AgentDB]:
    return OUR_NFT_AGENTS + list(AgentTableHandler().sql_handler.get_all())
