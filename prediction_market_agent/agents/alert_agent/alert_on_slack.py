import logging
import time
from datetime import datetime

import pandas as pd
import requests
from dune_client.client import DuneClient
from dune_client.query import QueryBase, QueryParameter
from prediction_market_agent_tooling.deploy.agent import DeployableAgent
from prediction_market_agent_tooling.markets.markets import MarketType

PERFORMANCE_QUERY_ID = 4244657
PRECISION_QUERY_ID = 5620557
MAX_WAIT_MINUTES = 30
POLL_INTERVAL_SECONDS = 30
SAFE_SLACK_MESSAGE_LENGTH = 500


class PerformanceAlertAgent(DeployableAgent):
    def load(self) -> None:
        self.slack_webhook: str = self.api_keys.slack_webhook_url.get_secret_value()
        self.dune = DuneClient(self.api_keys.dune_api_key.get_secret_value())

    def _get_performance_data_from_dune(
        self, market_type: MarketType
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        performance_query = self._querry(
            PERFORMANCE_QUERY_ID,
            "Agents Performance For last week",
            "Last Month",
            "PMA",
        )
        precision_query = self._querry(
            PRECISION_QUERY_ID, "Agents Precision For last week", "Last Month", "PMA"
        )

        performance_result = self._get_latest_result(performance_query.query_id)
        if performance_result is None:
            performance_result = self._process_new_querry(performance_query)

        precision_result = self._get_latest_result(precision_query.query_id)
        if precision_result is None:
            precision_result = self._process_new_querry(precision_query)

        return performance_result, precision_result

    def _wait_for_dune_execution(self, query_id: int) -> pd.DataFrame | None:
        start_time = time.time()
        while time.time() < start_time + MAX_WAIT_MINUTES * 60:
            res = self._get_latest_result(query_id)
            if res:
                return pd.DataFrame(res.get("rows", []))
            time.sleep(POLL_INTERVAL_SECONDS)
        res = self._get_latest_result(query_id)
        if res:
            return res
        raise RuntimeError(f"No result available for query {query_id}")

    def _process_new_querry(self, query: QueryBase) -> pd.DataFrame | None:
        try:
            res = self.dune.run_query(query, ping_frequency=POLL_INTERVAL_SECONDS)
            if res.result and res.result.rows:
                return pd.DataFrame(res.result.rows)
            return None
        except Exception:
            logging.error(
                f"Querry timed out {query.query_id}, waiting for {MAX_WAIT_MINUTES} minutes for it to finish"
            )
            return self._wait_for_dune_execution(query.query_id)

    def _get_latest_result(self, query_id: int) -> pd.DataFrame | None:
        try:
            res = self.dune.get_latest_result(query_id, max_age_hours=8)
            if res.result and res.result.rows:
                return pd.DataFrame(res.result.rows)
            return None
        except Exception:
            return None

    def _querry(
        self, query_id: int, name: str, range_value: str, ranking_value: str
    ) -> QueryBase:
        return QueryBase(
            name=name,
            query_id=query_id,
            params=[
                QueryParameter.text_type(name="range", value=range_value),
                QueryParameter.text_type(name="ranking", value=ranking_value),
            ],
        )

    def _weekly_cumulative_profit(self, df_performance: pd.DataFrame) -> pd.DataFrame:
        df_performance["block_date"] = pd.to_datetime(
            df_performance["block_date"]
        ).dt.tz_localize(None)
        df_performance = df_performance.sort_values(
            ["label", "block_date"]
        ).reset_index(drop=True)

        # Collect weekly-picked rows for all labels
        rows: list[pd.DataFrame] = []
        for _, label_group in df_performance.groupby("label"):
            label_group = label_group.sort_values("block_date").reset_index(drop=True)

            idx = pd.DatetimeIndex(label_group["block_date"].values)
            if idx.empty:
                continue

            # Create weekly "anchor" dates by walking backward from the most recent date
            anchors: list[pd.Timestamp] = []
            cur = idx.max()
            while cur >= idx.min():
                anchors.append(cur)
                cur -= pd.Timedelta(days=7)
            if not anchors:
                continue

            pos = idx.get_indexer(pd.DatetimeIndex(anchors), method="ffill")
            mask = pos != -1
            if not mask.any():
                continue

            picked = pd.DatetimeIndex(idx[pos[mask]]).unique().sort_values()
            label_group["date_only"] = label_group["block_date"].dt.date
            picked_df = label_group[
                label_group["date_only"].isin(set(picked.date))
            ].copy()

            if len(picked_df) > 0:
                picked_df = picked_df.drop("date_only", axis=1)
                picked_df = picked_df[
                    ["label", "block_date", "cumulative_return"]
                ].copy()
                rows.append(picked_df)

        if not rows:
            return pd.DataFrame(columns=["label", "block_date", "cumulative_return"])

        result = (
            pd.concat(rows, ignore_index=True)
            .drop_duplicates(subset=["label", "block_date"])
            .sort_values(["label", "block_date"])
            .reset_index(drop=True)
        )

        result["cumulative_return"] = pd.to_numeric(
            result["cumulative_return"], errors="coerce"
        )
        result["block_date"] = pd.to_datetime(
            result["block_date"], utc=True
        ).dt.normalize()
        result = result.sort_values("block_date").reset_index(drop=True)
        result["label"] = result["label"].astype(str)

        return result

    def _prepare_precision_daily(self, df_precision: pd.DataFrame) -> pd.DataFrame:
        df_prec = df_precision.copy()
        if "label" not in df_prec.columns and "agent" in df_prec.columns:
            df_prec = df_prec.rename(columns={"agent": "label"})
        df_prec["date"] = pd.to_datetime(df_prec["day"], utc=True).dt.normalize()
        df_prec["pct_correct"] = pd.to_numeric(
            df_prec.get("pct_correct"), errors="coerce"
        )
        df_prec["cnt_resolved"] = pd.to_numeric(
            df_prec.get("cnt_resolved"), errors="coerce"
        )
        df_prec = df_prec.sort_values(["label", "date"]).reset_index(drop=True)

        rolled = (
            df_prec.set_index("date")
            .groupby("label")["pct_correct"]
            .rolling("7D", min_periods=1)
            .mean()
            .rename("pct_correct_7d")
            .reset_index()
        )

        df_prec = df_prec.merge(rolled, on=["label", "date"], how="left")

        result = df_prec[
            ["label", "date", "pct_correct", "pct_correct_7d", "cnt_resolved"]
        ]
        result = (
            result[["label", "date", "pct_correct", "pct_correct_7d", "cnt_resolved"]]
            .dropna(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        result["label"] = result["label"].astype(str)
        return result

    @staticmethod
    def format_profit(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{float(v):.2f}$"

    @staticmethod
    def format_daily_accuracy(v: float) -> str:
        if pd.isna(v):
            return "n/a"
        v = float(v)
        return f"{v:.1f}%" if v > 1 else f"{v:.1%}"

    @staticmethod
    def format_avg_accuracy(v: float) -> str:
        if pd.isna(v):
            return "n/a"
        v = float(v)
        return f"{v:.1f}%" if v > 1 else f"{v:.1%}"

    @staticmethod
    def format_count(v: float) -> str:
        return "n/a" if pd.isna(v) else f"{int(max(float(v), 0))}"

    def _get_performance_report(
        self, performance_df: pd.DataFrame, precision_df: pd.DataFrame
    ) -> pd.DataFrame:
        weekly_cumulative_profit = self._weekly_cumulative_profit(performance_df)
        accuracy = self._prepare_precision_daily(precision_df)

        # Merge weekly performance with accuracy
        weekly_combined = pd.merge_asof(
            weekly_cumulative_profit,
            accuracy,
            left_on="block_date",
            right_on="date",
            by="label",
            direction="nearest",
            tolerance=pd.Timedelta("3D"),
        )
        weekly_combined = weekly_combined.drop(columns=["date"])
        weekly_combined["daily_pct_correct"] = weekly_combined["pct_correct"]
        weekly_combined["precision_7d_avg"] = weekly_combined["pct_correct_7d"]
        weekly_combined["daily_cnt_resolved"] = weekly_combined["cnt_resolved"]
        weekly_combined = weekly_combined.drop(
            columns=["pct_correct", "pct_correct_7d", "cnt_resolved"]
        )
        weekly_combined["sequence"] = weekly_combined.groupby("label").cumcount() + 1

        # Format columns
        weekly_combined["profit_formatted"] = weekly_combined[
            "cumulative_return"
        ].apply(self.format_profit)
        weekly_combined["daily_accuracy_formatted"] = weekly_combined[
            "daily_pct_correct"
        ].apply(self.format_daily_accuracy)
        weekly_combined["avg_accuracy_formatted"] = weekly_combined[
            "precision_7d_avg"
        ].apply(self.format_avg_accuracy)
        weekly_combined["count_formatted"] = weekly_combined[
            "daily_cnt_resolved"
        ].apply(self.format_count)
        return weekly_combined

    @staticmethod
    def _trend_icon(prev_val: float | None, cur_val: float | None) -> str:
        if prev_val is None or cur_val is None:
            return ""
        if cur_val > prev_val:
            return "📈"
        if cur_val < prev_val:
            return "📉"
        return "➡️"

    @staticmethod
    def _accuracy_icon(val_str: str) -> str:
        if val_str == "n/a":
            return ""
        try:
            if "%" in val_str:
                num = float(val_str.replace("%", ""))
            else:
                num = float(val_str) * 100.0
            if num >= 70:
                return "🟢"
            if num >= 50:
                return "🟡"
            return "🔴"
        except Exception:
            return ""

    def _format_agent_block(self, label: str, g: pd.DataFrame) -> str:
        lines: list[str] = [f"{str(label)}"]
        prev_profit_num: float | None = None

        for _, r in g.iterrows():
            date_str = (
                r["block_date"].strftime("%Y-%m-%d")
                if pd.notna(r["block_date"])
                else "N/A"
            )

            cur_profit_num = (
                float(r["cumulative_return"])
                if pd.notna(r["cumulative_return"])
                else None
            )
            trend = self._trend_icon(prev_profit_num, cur_profit_num)
            profit_str = (
                f"{trend}{r['profit_formatted']}"
                if trend
                else f"{r['profit_formatted']}"
            )
            prev_profit_num = cur_profit_num

            daily_acc = str(r.get("daily_accuracy_formatted", "n/a"))
            avg_acc = str(r.get("avg_accuracy_formatted", "n/a"))
            daily_i = self._accuracy_icon(daily_acc)
            avg_i = self._accuracy_icon(avg_acc)
            if daily_acc != "n/a" and avg_acc != "n/a":
                acc_str = f"{daily_i}{daily_acc} (avg:{avg_i}{avg_acc})"
            elif daily_acc != "n/a":
                acc_str = f"{daily_i}{daily_acc}"
            else:
                acc_str = "n/a"

            count_fmt = str(r.get("count_formatted", "n/a"))
            count_str = f"{count_fmt} resolved" if count_fmt != "n/a" else "n/a"

            lines.append(f"  '{date_str}' | {count_str} | {profit_str} | {acc_str}")

        lines.append("")  # spacer
        return "\n".join(lines)

    def _prepare_report_string(
        self, combined_report: pd.DataFrame
    ) -> tuple[str, list[str]]:
        combined_report["block_date"] = pd.to_datetime(
            combined_report["block_date"], utc=True, errors="coerce"
        )
        combined_report = combined_report.sort_values(
            ["label", "block_date", "sequence"]
        ).reset_index(drop=True)

        total_agents = combined_report["label"].nunique()
        total_weeks = combined_report["sequence"].nunique()
        header = (
            f"*AGENT PERFORMANCE REPORT*\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Total Agents: {total_agents}\n"
            f"Total Weeks: {total_weeks}\n\n"
            f"Performance data grouped by agent...\n"
            f"Legend:\n"
            f"Date | Resolved Count (>0 or 0) | Profit (trends: 📈📉➡️) | "
            f"Accuracy (🟢≥70% 🟡50-79% 🔴<50%)\n\n"
        )

        body = [
            self._format_agent_block(label, g)
            for label, g in combined_report.groupby("label", sort=False)
        ]
        return header, body

    def _send_report_to_slack(self, header: str, body: list[str]) -> None:
        # Send header first
        if header:
            requests.post(self.slack_webhook, json={"text": header}, timeout=10)

        # Split body into agent blocks (separated by double newlines)
        if body:
            current_message = ""

            for block in body:
                block_with_spacing = block + "\n\n"  # restore spacing

                # If adding this agent block would exceed limit, send current message first
                if (
                    current_message
                    and len(current_message) + len(block_with_spacing)
                    > SAFE_SLACK_MESSAGE_LENGTH
                ):
                    requests.post(
                        self.slack_webhook,
                        json={"text": current_message.rstrip()},
                        timeout=10,
                    )
                    current_message = block_with_spacing
                else:
                    current_message += block_with_spacing

            # Send any remaining content
            if current_message:
                requests.post(
                    self.slack_webhook,
                    json={"text": current_message.rstrip()},
                    timeout=10,
                )

    def run(self, market_type: MarketType) -> None:
        performance_df, precision_df = self._get_performance_data_from_dune(market_type)
        combined_report = self._get_performance_report(performance_df, precision_df)
        header, body = self._prepare_report_string(combined_report)
        self._send_report_to_slack(header, body)
