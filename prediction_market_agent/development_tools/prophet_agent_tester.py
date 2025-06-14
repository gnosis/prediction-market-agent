import json
import time
from functools import partial

import pandas as pd
import tenacity
from eth_typing import HexAddress, HexStr
from prediction_market_agent_tooling.benchmark.utils import Prediction
from prediction_market_agent_tooling.deploy.betting_strategy import BettingStrategy
from prediction_market_agent_tooling.gtypes import Probability
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import (
    USD,
    CategoricalProbabilisticAnswer,
    ProbabilisticAnswer,
    ResolvedBet,
    Trade,
)
from prediction_market_agent_tooling.markets.omen.omen import OmenAgentMarket
from prediction_market_agent_tooling.markets.omen.omen_contracts import (
    OmenConditionalTokenContract,
)
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    get_omen_market_by_market_id_cached,
)
from prediction_market_agent_tooling.tools.transaction_cache import (
    TransactionBlockCache,
)
from prediction_prophet.autonolas.research import Prediction as PredictionProphet
from prediction_prophet.functions.research import Research
from pydantic import BaseModel
from pydantic_ai import Agent
from web3.exceptions import TransactionNotFound


class ProphetTestResult(BaseModel):
    run_name: str
    market_question: str
    research: Research
    prediction: Prediction
    trades: list[Trade]
    market_resolution: str
    profit_usd: USD


class ProphetTestMetrics(BaseModel):
    total_trades: int
    binary_prediction_accuracy: float
    weighted_prediction_accuracy: float
    prediction_brier_score: float
    binary_trade_accuracy: float | None
    investment_usd: USD | None
    profit_usd: USD | None
    roi: float | None


class ProphetAgentTester:
    def __init__(
        self,
        prophet_research: partial[Research],
        prophet_predict: partial[PredictionProphet],
        betting_strategy: BettingStrategy | None = None,
        use_old_research: bool = False,
        use_old_prediction: bool = False,
        mocked_agent_name: str = "DeployablePredictionProphetGPT4oAgent",
        max_trades_to_test_on: int = 10,
        run_name: str = "test_prophet_agent_baseline",
        delay_between_trades: float = 0.5,
        simulate_trades: bool = True,
        bet_only: bool = True,
        only_xdai_bets: bool = True,
    ):
        self.prophet_research = prophet_research
        self.prophet_predict = prophet_predict
        self.betting_strategy = betting_strategy
        self.max_trades_to_test_on = max_trades_to_test_on
        self.mocked_agent_name = mocked_agent_name
        self.use_old_research = use_old_research
        self.use_old_prediction = use_old_prediction
        self.run_name = run_name
        self.delay_between_trades = delay_between_trades

        self.simulate_trades = simulate_trades
        self.tx_block_cache = TransactionBlockCache(
            web3=OmenConditionalTokenContract().get_web3()
        )
        self.bet_only = bet_only
        self.only_xdai_bets = only_xdai_bets

    def test_prophet_agent(
        self, dataset: pd.DataFrame, research_agent: Agent, prediction_agent: Agent
    ) -> list[ProphetTestResult]:
        filtered_dataset = dataset[dataset["agent_name"] == self.mocked_agent_name]
        available_trades = len(filtered_dataset)
        trades_to_process = min(self.max_trades_to_test_on, available_trades)

        logger.info(
            f"Found {available_trades} trades for {self.mocked_agent_name}, processing {trades_to_process}"
        )

        results = []

        if self.bet_only:
            filtered_dataset = filtered_dataset[filtered_dataset["bet_json"].notna()]

        filtered_dataset = filtered_dataset.head(self.max_trades_to_test_on)

        for index, (_, item) in enumerate(filtered_dataset.iterrows()):
            logger.info(
                f"Processing trade {index + 1}/{trades_to_process}: {item['market_question']}"
            )

            if self.simulate_trades and self.delay_between_trades > 0:
                time.sleep(self.delay_between_trades)

            market = OmenAgentMarket.model_validate(
                json.loads(item["full_market_json"])
            )
            try:
                trades, prediction, research = self.execute_prophet_partials(
                    market=market,
                    research_output=item["analysis"],
                    prediction_output=item["prediction_json"],
                    research_agent=research_agent,
                    prediction_agent=prediction_agent,
                )

                if self.simulate_trades:
                    bet = ResolvedBet.model_validate_json(item["bet_json"])

                    try:
                        bet_tx_block_number = self.tx_block_cache.get_block_number(
                            bet.id
                        )
                    except tenacity.RetryError as e:
                        if isinstance(e.last_attempt.exception(), TransactionNotFound):
                            logger.warning(
                                f"Transaction not found for trace {item['trace_id']} and bet {bet.id}"
                            )

                    market_before_placing_bet = get_omen_market_by_market_id_cached(
                        HexAddress(HexStr(market.id)),
                        block_number=bet_tx_block_number - 1,
                    )
                    omen_agent_market_before_placing_bet = (
                        OmenAgentMarket.from_data_model(market_before_placing_bet)
                    )

                    buy_trade_in_tokes = (
                        omen_agent_market_before_placing_bet.get_in_token(
                            trades[0].amount
                        )
                    )
                    probs = prediction.outcome_prediction.probabilities  # type: ignore
                    predicted = max(probs, key=lambda k: probs[k])
                    actual_resolution = item["market_resolution"].lower()

                    if trades[0].outcome.lower() != actual_resolution:
                        profit_outcome_token = -buy_trade_in_tokes

                    else:
                        received_outcome_tokens = (
                            omen_agent_market_before_placing_bet.get_buy_token_amount(
                                trades[0].amount, outcome=trades[0].outcome
                            )
                        )
                        profit_outcome_token = (
                            received_outcome_tokens.as_token - buy_trade_in_tokes
                        )

                profit_usd = omen_agent_market_before_placing_bet.get_token_in_usd(
                    profit_outcome_token
                )
                test_result = ProphetTestResult(
                    run_name=self.run_name,
                    market_question=item["market_question"],
                    research=research,
                    prediction=prediction,
                    trades=trades,
                    market_resolution=item["market_resolution"],
                    profit_usd=profit_usd,
                )
                results.append(test_result)
            except Exception as e:
                logger.error(f"Error processing trade {index + 1}: {e}")
                continue
        logger.info(
            f"Completed processing {len(results)} trades for {self.mocked_agent_name}"
        )
        return results

    def to_research_output(self, research_output: str) -> Research:
        return Research(
            report=research_output,
            all_queries=[],
            reranked_queries=[],
            websites_to_scrape=[],
            websites_scraped=[],
        )

    def to_prediction_output(self, prediction_output: str) -> PredictionProphet:
        prediction = json.loads(prediction_output)
        return PredictionProphet(
            decision="y" if prediction["p_yes"] > 0.5 else "n",
            p_yes=prediction["p_yes"],
            p_no=prediction["p_no"],
            confidence=prediction["confidence"],
            info_utility=prediction["info_utility"],
            reasoning=prediction["reasoning"],
            logprobs=None,
        )

    def execute_prophet_partials(
        self,
        market: AgentMarket,
        research_output: str,
        prediction_output: str,
        research_agent: Agent,
        prediction_agent: Agent,
    ) -> tuple[list[Trade], Prediction, Research]:
        research = (
            self.prophet_research(research_agent, market.question, market)
            if not self.use_old_research
            else self.to_research_output(research_output)
        )
        prediction_prophet: PredictionProphet = (
            self.prophet_predict(
                prediction_agent, market.question, research.report, market
            )
            if not self.use_old_prediction
            else self.to_prediction_output(prediction_output)
        )

        probabilistic_answer = ProbabilisticAnswer(
            p_yes=Probability(prediction_prophet.p_yes),
            reasoning=prediction_prophet.reasoning,
            confidence=prediction_prophet.confidence,
        )

        prediction = Prediction(
            outcome_prediction=CategoricalProbabilisticAnswer.from_probabilistic_answer(
                probabilistic_answer
            )
        )

        trades = []
        if (
            self.simulate_trades
            and prediction.outcome_prediction
            and self.betting_strategy
        ):
            try:
                trades = self.betting_strategy.calculate_trades(
                    None,
                    prediction.outcome_prediction,
                    market,
                )
            except Exception as e:
                logger.error(f"Error calculating trades: {e}")

        return trades, prediction, research

    def evaluate_results(
        self,
        test_results: list[ProphetTestResult],
        print_individual_metrics: bool = False,
    ) -> ProphetTestMetrics | None:
        if not test_results:
            logger.warning("No test results to evaluate")
            return None

        total_trades = len(test_results)
        logger.info(f"Evaluating {total_trades} test results")

        # Filter out results with no outcome prediction - do this once
        valid_results = [
            result
            for result in test_results
            if (
                result.prediction.outcome_prediction is not None
                and result.prediction.outcome_prediction.probabilities is not None
            )
        ]

        y_true = [result.market_resolution.lower() for result in valid_results]

        y_pred_label = []
        y_pred_weight = []
        for result in valid_results:
            probs = result.prediction.outcome_prediction.probabilities  # type: ignore
            max_outcome = max(probs, key=lambda k: probs[k])
            y_pred_label.append(max_outcome.lower())
            y_pred_weight.append(probs[max_outcome])

        binary_prediction_accuracy = [
            1 if true_val == pred_val else 0
            for true_val, pred_val in zip(y_true, y_pred_label)
        ]

        weighted_prediction_accuracy = [
            weight if true_val == pred_val else (1 - weight)
            for true_val, pred_val, weight in zip(y_true, y_pred_label, y_pred_weight)
        ]

        brier_scores = []
        for result in valid_results:
            actual_outcome = result.market_resolution.lower()

            if not result.prediction.outcome_prediction:
                continue

            probs = dict(result.prediction.outcome_prediction.probabilities)

            brier_score = 0.0
            for outcome, predicted_prob in probs.items():
                true_value = 1.0 if outcome.lower() == actual_outcome else 0.0
                brier_score += (predicted_prob - true_value) ** 2

            brier_scores.append(brier_score)

        avg_brier_score = sum(brier_scores) / len(brier_scores) if brier_scores else 0

        avg_binary_prediction_accuracy = sum(binary_prediction_accuracy) / len(
            binary_prediction_accuracy
        )
        avg_weighted_prediction_accuracy = sum(weighted_prediction_accuracy) / len(
            weighted_prediction_accuracy
        )

        avg_binary_trade_accuracy = None
        if self.simulate_trades:
            y_trade_outcome = [
                result.trades[0].outcome.lower() if result.trades else None
                for result in valid_results
            ]
            trade_investments = [
                result.trades[0].amount for result in valid_results if result.trades
            ]
            binary_trade_accuracy = [
                1 if true_val == trade_val else 0
                for true_val, trade_val in zip(y_true, y_trade_outcome)
                if trade_val is not None
            ]
            avg_binary_trade_accuracy = (
                sum(binary_trade_accuracy) / len(binary_trade_accuracy)
                if binary_trade_accuracy
                else 0
            )

        sum_profit_usd = sum(result.profit_usd for result in valid_results)
        sum_investment_usd = sum(trade_investments)

        metrics = ProphetTestMetrics(
            total_trades=total_trades,
            binary_prediction_accuracy=avg_binary_prediction_accuracy,
            weighted_prediction_accuracy=avg_weighted_prediction_accuracy,
            binary_trade_accuracy=avg_binary_trade_accuracy,
            prediction_brier_score=avg_brier_score,
            profit_usd=sum_profit_usd if sum_profit_usd else None,
            investment_usd=sum_investment_usd if sum_investment_usd else None,
            roi=(
                ((sum_profit_usd) / sum_investment_usd) * 100
                if sum_investment_usd and sum_profit_usd
                else None
            ),
        )

        if print_individual_metrics:
            logger.info("\n" + "=" * 50)
            logger.info(f"EVALUATION METRICS {self.run_name}")
            logger.info("=" * 50)
            logger.info(f"Total Trades: {total_trades}")
            logger.info(
                f"Binary Prediction Accuracy: {avg_binary_prediction_accuracy:.4f}"
            )
            logger.info(
                f"Weighted Prediction Accuracy: {avg_weighted_prediction_accuracy:.4f}"
            )
            logger.info(f"Brier Score: {avg_brier_score:.4f}")

            if self.simulate_trades:
                logger.info(f"Binary Trade Accuracy: {avg_binary_trade_accuracy:.4f}")
                logger.info(f"Profit USD: {metrics.profit_usd}")
                logger.info(f"Investment USD: {metrics.investment_usd}")
                logger.info(f"ROI: {metrics.roi:.4f}%")

        return metrics
