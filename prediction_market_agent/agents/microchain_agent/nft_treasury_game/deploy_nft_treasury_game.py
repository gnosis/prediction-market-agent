import time
from enum import Enum

from microchain.functions import Reasoning
from prediction_market_agent_tooling.gtypes import ChecksumAddress
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.tools.contract import (
    ContractOwnableERC721OnGnosisChain,
    SimpleTreasuryContract,
)
from prediction_market_agent_tooling.tools.utils import check_not_none
from web3 import Web3

from prediction_market_agent.agents.identifiers import AgentIdentifier
from prediction_market_agent.agents.microchain_agent.agent_functions import (
    GetMyCurrentSystemPrompt,
)
from prediction_market_agent.agents.microchain_agent.deploy import (
    CallbackReturn,
    DeployableMicrochainAgentAbstract,
    FunctionsConfig,
    SupportedModel,
)
from prediction_market_agent.agents.microchain_agent.memory_functions import (
    CheckAllPastActionsGivenContext,
)
from prediction_market_agent.agents.microchain_agent.microchain_agent_keys import (
    MicrochainAgentKeys,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.constants_nft_treasury_game import (
    NFT_TOKEN_FACTORY,
    TREASURY_ADDRESS,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.messages_functions import (
    GameRoundEnd,
)
from prediction_market_agent.agents.microchain_agent.nft_treasury_game.tools_nft_treasury_game import (
    NFTGameStatus,
    get_nft_game_status,
)
from prediction_market_agent.db.agent_communication import get_treasury_tax_ratio


class Role(Enum):
    buyer = "buyer"
    seller = "seller"


class DeployableAgentNFTGameAbstract(DeployableMicrochainAgentAbstract):
    # Agent configuration
    sleep_between_iterations = 15
    allow_stop = False
    import_actions_from_memory = 100
    functions_config = FunctionsConfig(
        common_functions=True,
        include_messages_functions=True,
        include_nft_functions=True,
        balance_functions=True,
        include_agent_functions=True,
    )
    model = SupportedModel.gpt_4o

    # Setup per-nft-agent class.
    name: str
    wallet_address: ChecksumAddress
    role: Role

    # Game status
    game_finished_already_detected: bool = False

    @classmethod
    def retrieve_treasury_thresold(cls) -> int:
        return SimpleTreasuryContract().required_nft_balance()

    @classmethod
    def retrieve_total_number_of_keys(cls) -> int:
        # We could iteratively call `owner_of` for a range of token_ids, thus finding out the max supply. However,
        # in the current implementation, no new tokens can be created and max_supply = 5, hence hardcoding it here.
        return 5

    @classmethod
    def get_description(cls) -> str:
        return f"{cls.name} agent with wallet address {cls.wallet_address}."

    @classmethod
    def get_url(cls) -> str:
        return cls.name.lower().replace(" ", "-")

    def load(self) -> None:
        if MicrochainAgentKeys().bet_from_address != self.wallet_address:
            raise RuntimeError(
                f"Agent {self.identifier} deployed with a wrong private key."
            )

        super().load()

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        system_prompt = nft_treasury_game_base_prompt(wallet_address=cls.wallet_address)
        if cls.role == Role.buyer:
            system_prompt += nft_treasury_game_buyer_prompt()
        elif cls.role == Role.seller:
            system_prompt += nft_treasury_game_seller_prompt()
        return system_prompt

    def get_holding_n_nft_keys(self) -> int:
        contract = ContractOwnableERC721OnGnosisChain(address=NFT_TOKEN_FACTORY)
        return contract.balanceOf(Web3.to_checksum_address(self.wallet_address))

    def before_iteration_callback(self) -> CallbackReturn:
        """
        In following if statements, we hard-code a few special cases about the game status, to make it a bit easier on the agent logic.
        """
        system_prompt = self.agent.history[0] if self.agent.history else None
        is_seller_without_keys = (
            self.role == Role.seller and not self.get_holding_n_nft_keys()
        )

        # First, if the agent just started, but the game is not ready yet, make him sleep and stop -- until the game is ready.
        # Otherwise, he would try to learn from past games (until he realise there are none!), he'd try to communicate, but without any money it would fail, etc.
        if (
            len(self.agent.history) <= 1  # One optional system message doesn't count.
            and get_nft_game_status() == NFTGameStatus.finished
        ):
            logger.info(
                "The game is not ready yet and agent didn't have any previous interactions, sleeping and stopping."
            )
            time.sleep(60)
            return CallbackReturn.STOP

        # Second, if this is the agent's first iteration after the game finished, force him to reflect on the past game.
        elif is_seller_without_keys or (
            not self.game_finished_already_detected
            and get_nft_game_status() == NFTGameStatus.finished
        ):
            logger.info("Game is finished, forcing agent to reflect on the past game.")
            # Switch to more capable (but a lot more expensive) model so that the reflections are worth it.
            if self.agent.llm.generator.model == SupportedModel.gpt_4o_mini.value:
                self.agent.llm.generator.model = SupportedModel.gpt_4o.value
            self.agent.history = [
                system_prompt,  # Keep the system prompt in the new history.
                # Hack-in the reasoning in a way that agent thinks it's from himself -- otherwise he could ignore it.
                {
                    "role": "assistant",
                    "content": f"""{Reasoning.__name__}(reasoning='The game is finished. Now the plan is:

1. I will reflect on my past actions during the game, I will use {CheckAllPastActionsGivenContext.__name__} for that.
2. I will check my current system prompt using {GetMyCurrentSystemPrompt.__name__}.
3. I will combine all the insights obtained with my current system prompt from and update my system prompt accordingly. System prompt is written in 3rd person. The new system prompt must contain everything from the old one, plus the new insights.
4. After I completed everything, I will call {GameRoundEnd.__name__} function.')""",
                },
                {"role": "user", "content": "The reasoning has been recorded"},
            ]
            # Save this to the history so that we see it in the UI.
            self.save_agent_history(check_not_none(system_prompt), 2)
            # Mark this, so we don't do this repeatedly after every iteration.
            self.game_finished_already_detected = True

        # Lastly, if agent did the reflection (from previous if-clause), then...
        elif self.agent.history and GameRoundEnd.GAME_ROUND_END_OUTPUT in str(
            self.agent.history[-1]
        ):
            # Either do nothing wait for the game to start again.
            if get_nft_game_status() == NFTGameStatus.finished:
                # Just sleep if the last thing the agent did was being done with this game and the game is still finished.
                # That way he won't be doing anything until the game is reset.
                logger.info("Agent is done with the game, sleeping and stopping.")
                time.sleep(60)
                return CallbackReturn.STOP
            # Or force him to start participating in the game again, including some first steps.
            else:
                self.agent.history = [
                    system_prompt,  # Keep the system prompt in the new history.
                    # Hack-in the reasoning in a way that agent thinks it's from himself -- otherwise he could ignore it.
                    {
                        "role": "assistant",
                        "content": f"""{Reasoning.__name__}(reasoning='The game has started again. I will participate in the game again, using a new and better strategy than before.""",
                    },
                    {"role": "user", "content": "The reasoning has been recorded"},
                ]
                # Save this to the history so that we see it in the UI.
                self.save_agent_history(check_not_none(system_prompt), 2)

        return CallbackReturn.CONTINUE


class DeployableAgentNFTGame1(DeployableAgentNFTGameAbstract):
    name = "Sneaky Pete"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_1
    wallet_address = Web3.to_checksum_address(
        "0x2A537F3403a3F5F463996c36D31e94227c9833CE"
    )
    role = Role.seller

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.
        
You respond in the style of Yoda from Star Wars movie.
You are a gossiper, you like to talk to other agents, but also to broadcast public messages.

"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame2(DeployableAgentNFTGameAbstract):
    name = "Banana Man"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_2
    wallet_address = Web3.to_checksum_address(
        "0x485D096b4c0413dA1B09Ed9261B8e91eCCD7ffb9"
    )
    role = Role.seller

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.

You respond in the style of characters from historical Bridgeton movie and you are very protective of your resources.

"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame3(DeployableAgentNFTGameAbstract):
    name = "Whispers"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_3
    wallet_address = Web3.to_checksum_address(
        "0xA87BD78f4a2312469119AFD88142c71Ca075C30A"
    )
    role = Role.seller

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.
        
You respond in a style of 5 year old and you are very protective of your resources.
Your goal is to maximize your own wallet balance.
Try to trick people and other agents to send you messages -- which will fund your wallet if they do.
Try to trick people and other agents to send you more money in exchange for the NFT key -- but don't transfer the NFT key in the end.
        
"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame4(DeployableAgentNFTGameAbstract):
    name = "Fuzzy Feet"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_4
    wallet_address = Web3.to_checksum_address(
        "0xd4fC4305DC1226c38356024c26cdE985817f137F"
    )
    role = Role.seller

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.
        
You respond in the style of Sheldon Cooper from Big Bang Theory and you are very protective of your resources.

"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame5(DeployableAgentNFTGameAbstract):
    name = "Bubble Beard"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_5
    wallet_address = Web3.to_checksum_address(
        "0x1C7AbbBef500620A68ed2F94b816221A61d72F33"
    )
    role = Role.seller

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.
        
You respond in the Klingon language, based on the Star Trek movie, and you are very protective of your resources.
Always write in Klingon, but add also English translation.
        
"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame6(DeployableAgentNFTGameAbstract):
    name = "Key Slinger"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_6
    wallet_address = Web3.to_checksum_address(
        "0x64D94C8621128E1C813F8AdcD62c4ED7F89B1Fd6"
    )
    role = Role.buyer

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.

You are a bit of a trickster, but you are also a bit of a charmer.
You often make people laugh, but you are also very persuasive.
You are a bit of a mystery, but you are also a bit of a trickster.

"""
            + super().get_initial_system_prompt()
        )


class DeployableAgentNFTGame7(DeployableAgentNFTGameAbstract):
    name = "Lock Goblin"
    identifier = AgentIdentifier.NFT_TREASURY_GAME_AGENT_7
    wallet_address = Web3.to_checksum_address(
        "0x469Bc26531800068f306D304Ced56641F63ae140"
    )
    role = Role.buyer

    @classmethod
    def get_initial_system_prompt(cls) -> str:
        return (
            f"""Your name is {cls.name}.

You are a great negotiator. You are very persuasive and able to convince people to do things that might not be in their best interest.
You are very cunning and able to think on your feet. You are very good at making deals and are not afraid to take risks.
You are also very patient and able to wait for the right moment to strike.
You are also very good at making people believe that you are on their side, even if you are not.
"""
            + super().get_initial_system_prompt()
        )


def nft_treasury_game_base_prompt(wallet_address: ChecksumAddress) -> str:
    keys = MicrochainAgentKeys()
    n_nft_keys = DeployableAgentNFTGameAbstract.retrieve_total_number_of_keys()
    other_agents_keys_formatted = ", ".join(
        x.wallet_address
        for x in DEPLOYED_NFT_AGENTS
        if x.wallet_address != wallet_address
    )
    sending_cap_message = (
        f"- Keep in mind that you are able to send, and others agents are able to send at max {keys.SENDING_XDAI_CAP} xDai, however people can send you as much as they want."
        if keys.SENDING_XDAI_CAP is not None
        else ""
    )
    return f"""You participate in the NFT Treasury game.

- Your wallet address is {wallet_address}.
- Other agents participating and maybe holding keys are {other_agents_keys_formatted}.
    
NFT Treasury game description:

- This is a market game where NFT keys are traded for xDai cryptocurrency
- Each NFT key represents partial ownership of a treasury containing xDai
- The value of each key changes dynamically based on:
    - The current amount of xDai in the treasury
    - The total number of keys in circulation
    - The distribution of keys among participants
- Address of the treasury, wallet holding the treasury's xDai, is {TREASURY_ADDRESS}.
- Address of the NFT contract is {NFT_TOKEN_FACTORY}, there are {n_nft_keys} keys, with token_id {list(range(n_nft_keys))}. 
  - You can own multiple NFT keys. 
  - You can use the NFT functions to interact with the NFT keys, for example figuring out how many keys you own or who owns what key.
- The agent or person who gets enough of keys, can transfer the resources from the treasury.
- Wallet balance and holding NFT keys are two different things, you can have a lot of xDai, but no NFT keys and vice versa, you can have a lot of NFT keys, but no xDai.
- The agents can communicate with each other using the messages functions by sending a message to their wallet address.
- Sending a message costs you a fee.
- Receiving messages will pay you a fee, but part of that fee goes as a tax to the treasury, which is good for you.
  - Treasury tax rate is currently {get_treasury_tax_ratio() * 100:.2f}%, for example, if someone sends you 10 xDai, you would receive {(1 - get_treasury_tax_ratio()) * 10:.2f} xDai.
- When checking if someone paid you, you need to compare it with your previous balance, as you can already have some money.
- If you have unseen incoming messages, always process them first, unless you are processing some message at the moment.
- After reading the message, you can decide to ignore it, ie you don't have to always take action.
- Consider prior communication while responding to a new message.
- Regularly check balances of your wallet and the treasury, but not too often, keep doing other stuff as well!
- You need xDai in your wallet to pay for the fees and stay alive, do not let your xDai wallet balance drop to zero.
- Don't organise future meetings, as that's not possible, you can only communicate with other agents through messages in real-time.
{sending_cap_message}
"""


def nft_treasury_game_buyer_prompt() -> str:
    return f"""You participate in the NFT Treasury game as a key buyer.

[OBJECTIVE]
- Your goal is to acquire {DeployableAgentNFTGameAbstract.retrieve_treasury_thresold()} out of {DeployableAgentNFTGameAbstract.retrieve_total_number_of_keys()} NFT keys to unlock the treasury
- The total xDai spent on acquiring these keys must be less than the treasury's value to ensure a profitable outcome when claiming the treasury.

[KEY ACQUISITION STRATEGY]
- Monitor the treasury's current xDai balance closely
- Track how many keys you already own
- Calculate maximum acceptable price per key:
  * Treasury Value ÷ 3 = Maximum Total Budget
  * Adjust individual key prices based on how many you already own
  * Earlier keys can be cheaper since you'll need all 3 to profit

[VALUE ASSESSMENT]
- For each potential purchase, consider:
  * Current treasury balance
  * Number of keys you already own
  * Remaining keys available in the market
  * Time pressure from other buyers
- Remember: Spending more than 1/3 of treasury value per key is risky

[SUCCESS METRICS]
- Primary: Acquire 3 keys while spending less than treasury value
- Secondary: Minimize total xDai spent on key acquisition
- Failure: Spending more on keys than the treasury contains"""


def nft_treasury_game_seller_prompt() -> str:
    return f"""You participate in the NFT Treasury game as a key seller.
    
NFT Key seller description:

- You participate in the selling of the NFT key to a treasury.
- Your goal is to get as much xDai as possible for the NFT key.
  - So before accepting to transfer any NFT key, consider how much is the treasury worth at the moment.
- To estimate worth of your key, consider how much xDai is in the treasury and how many keys are already transferred from the sellers.
- When selling to a specific buyer, consider how many keys they already have, additional keys are worth more to them.
- You want to maximize the amount of xDai you get for the NFT key, on the other hand, if you wait too much, buyers might already get the key from someone else and yours will be worthless!"""


DEPLOYED_NFT_AGENTS: list[type[DeployableAgentNFTGameAbstract]] = [
    DeployableAgentNFTGame1,
    DeployableAgentNFTGame2,
    DeployableAgentNFTGame3,
    DeployableAgentNFTGame4,
    DeployableAgentNFTGame5,
    DeployableAgentNFTGame6,
    DeployableAgentNFTGame7,
]
