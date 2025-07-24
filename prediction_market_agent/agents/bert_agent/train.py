import os
from collections import Counter
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import typer
from huggingface_hub import login
from pinatapy import PinataPy
from prediction_market_agent_tooling.gtypes import IPFSCIDVersion0
from prediction_market_agent_tooling.loggers import logger
from prediction_market_agent_tooling.markets.omen.omen_subgraph_handler import (
    FilterBy,
    OmenSubgraphHandler,
    SortBy,
)
from prediction_market_agent_tooling.tools.betting_strategies.kelly_criterion import (
    CollateralToken,
)
from prediction_market_agent_tooling.tools.utils import utc_datetime
from pydantic import BaseModel, ConfigDict
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.calibration import calibration_curve

# Calibration imports
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryStatScores,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from prediction_market_agent.utils import APIKeys

APP = typer.Typer(pretty_exceptions_enable=False)


class ApproxAverageBinaryMarketProfitability(BinaryStatScores):
    def __init__(
        self,
        threshold: float,
        report_avg_profit: bool = False,
        multidim_average: str = "global",
        ignore_index: int | None = None,
        validate_args: bool = True,
    ):
        super().__init__(
            threshold=threshold,
            multidim_average=multidim_average,
            ignore_index=ignore_index,
            validate_args=validate_args,
        )

        self._all_preds = []
        self._all_targets = []

        self.report_avg_profit = report_avg_profit
        self.market_p_yes = 0.5  # Assume agent bets right after market creation.
        self.max_bet = CollateralToken(1.0)

    def compute(self) -> torch.Tensor:
        preds = torch.cat(self._all_preds, dim=0)
        targets = torch.cat(self._all_targets, dim=0)

        total_profit = 0.0
        total_bets = 0

        for pred, target in zip(preds, targets):
            pred = pred.item()
            target = target.item()

            # Use sigmoid if pred is a logit
            if not (0.0 <= pred <= 1.0):
                prob = float(torch.sigmoid(pred))
            else:
                prob = float(pred)

            pred_target = 1 if prob > self.threshold else 0
            profit = (
                self.max_bet.value / self.market_p_yes - self.max_bet.value
                if pred_target == target
                else -self.max_bet.value
            )

            total_profit += profit

        return (
            (
                torch.tensor(total_profit / total_bets, device=preds.device)
                if total_bets > 0
                else torch.tensor(0.0, device=preds.device)
            )
            if self.report_avg_profit
            else torch.tensor(total_profit, device=preds.device)
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self._all_preds.append(preds.detach().cpu().flatten())
        self._all_targets.append(target.detach().cpu().flatten())
        super().update(preds, target)

    def reset(self) -> None:
        self._all_preds = []
        self._all_targets = []
        super().reset()

    def plot(self, ax=None):
        profit = self.compute().item()
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        ax.bar(["Avg Profit"], [profit])
        ax.set_ylabel("Average Profit per Trade")
        ax.set_title("ApproxBinaryMarketProfitability")
        return (fig, ax)


class DataItem(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str | list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor


class TextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> DataItem:
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return DataItem(
            text=text,
            input_ids=encoding["input_ids"].squeeze(0),
            attention_mask=encoding["attention_mask"].squeeze(0),
            label=torch.tensor(label, dtype=torch.float),
        )


def collate_fn(batch: list[DataItem]) -> DataItem:
    text = [item.text for item in batch]
    input_ids = torch.stack([item.input_ids for item in batch])
    attention_mask = torch.stack([item.attention_mask for item in batch])
    labels = torch.stack([item.label for item in batch])
    return DataItem(
        text=text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        label=labels,
    )


class MarketClassifier(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_name: str,
        lr: float,
        total_steps: int,
        warmup_steps: int,
        max_length: int,
        dropout_prob: float,
        logit_scale: float,
        temperature: float,
        threshold: float = 0.5,
        calibrator=None,  # New: Optional calibrator
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["calibrator"])

        self.max_length = max_length
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.threshold = threshold
        self.fixed_class = None

        hidden_size = self.bert.config.hidden_size
        self.fcs = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for i in range(1)]
        )
        self.classifier = torch.nn.Linear(hidden_size, 1)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.lr = lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.logit_scale = logit_scale
        self.temperature = temperature

        self.loss_fn = torch.nn.BCELoss()

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.train_f1 = BinaryF1Score()
        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()
        self.train_precision = BinaryPrecision()
        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()
        self.train_profits = ApproxAverageBinaryMarketProfitability(
            threshold=self.threshold
        )
        self.val_profits = ApproxAverageBinaryMarketProfitability(
            threshold=self.threshold
        )
        self.test_profits = ApproxAverageBinaryMarketProfitability(
            threshold=self.threshold
        )

        # Calibration
        self.calibrator = calibrator

    def forward(self, input_ids, attention_mask):
        if self.fixed_class is not None:
            return torch.tensor([float(self.fixed_class)] * len(input_ids)).to(
                self.device
            )

        # Use mean pooling over the token embeddings to obtain the question's embedding.
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Mask out padding tokens before mean
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        x = sum_embeddings / torch.clamp(sum_mask, min=1e-9)

        for fc in self.fcs:
            x = fc(x)
            x = torch.nn.functional.relu(x)
            x = self.dropout(x)

        logits = self.classifier(x).squeeze(-1)

        logits = logits * self.logit_scale
        logits = logits / self.temperature

        probs = torch.sigmoid(logits)
        # If calibrator is present, apply it (for batch)
        if self.calibrator is not None and not self.training:
            # Detach and move to cpu for sklearn
            probs_np = probs.detach().cpu().numpy()
            # Calibrator expects 1d array
            calibrated = self.calibrator.transform(probs_np.reshape(-1, 1)).flatten()
            return torch.tensor(calibrated, device=probs.device, dtype=probs.dtype)
        return probs

    def predict_text_p_yes(self, text: str) -> float:
        self.eval()
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            prob = self(input_ids, attention_mask)
            return prob.item()

    def predict_text_boolean(self, text: str) -> bool:
        return self.predict_text_p_yes(text) > self.threshold

    def step(self, batch: DataItem, stage: str):
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        targets = batch.label.to(self.device)
        probs = self(input_ids, attention_mask)

        loss = self.loss_fn(probs, targets)
        preds = probs > self.threshold

        acc = getattr(self, f"{stage}_acc")
        acc.update(preds.cpu(), targets.int().cpu())
        f1 = getattr(self, f"{stage}_f1")
        f1.update(preds.cpu(), targets.int().cpu())
        precision = getattr(self, f"{stage}_precision")
        precision.update(preds.cpu(), targets.int().cpu())
        recall = getattr(self, f"{stage}_recall")
        recall.update(preds.cpu(), targets.int().cpu())
        profits = getattr(self, f"{stage}_profits")
        profits.update(probs.cpu(), targets.int().cpu())

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc)
        self.log(f"{stage}_f1", f1)
        self.log(f"{stage}_precision", precision)
        self.log(f"{stage}_recall", recall)
        self.log(f"{stage}_profits", profits)

        # Log current learning rate if scheduler is present
        if self.trainer is not None and self.trainer.optimizers:
            optimizer = self.trainer.optimizers[0]
            if optimizer.param_groups:
                current_lr = optimizer.param_groups[0]["lr"]
                self.log(f"{stage}_lr", current_lr, prog_bar=True)

        return loss

    def training_step(self, batch: DataItem, batch_idx: int):
        return self.step(batch, "train")

    def validation_step(self, batch: DataItem, batch_idx: int):
        return self.step(batch, "val")

    def test_step(self, batch: DataItem, batch_idx: int):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def get_torch_device() -> torch.device:
    # If MPS is available, use CPU instead to avoid placeholder storage errors.
    if torch.backends.mps.is_available():
        return torch.device("cpu")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def upload_to_ipfs(path: str) -> IPFSCIDVersion0:
    api_keys = APIKeys()
    pinata = PinataPy(
        api_keys.pinata_api_key.get_secret_value(),
        api_keys.pinata_api_secret.get_secret_value(),
    )
    ipfs_hash = IPFSCIDVersion0(
        pinata.pin_file_to_ipfs(
            path,
            save_absolute_paths=False,
        )["IpfsHash"]
    )
    return ipfs_hash


def label_prop(labels: list[int]) -> dict[str, float]:
    return (
        {label: labels.count(label) / len(labels) for label in set(labels)}
        if labels
        else {}
    )


@APP.command()
def main(
    limit_data: int | None = None,
    max_epochs: int = 100,
) -> None:
    login(APIKeys().huggingface_token.get_secret_value())

    output_dir = os.path.join(os.path.dirname(__file__), "training_logs")
    os.makedirs(output_dir, exist_ok=True)

    seed = 42
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    device = get_torch_device()
    batch_size = 64
    calibration_method = "platt"  # "platt" or "isotonic"

    # 1. Load data
    # Since we have OFVChallenger, to have reasonable accuracy in labels.
    start_date = utc_datetime(year=2024, month=10, day=1)
    # Time span of data in the test/val sets.
    val_span = timedelta(weeks=6)
    test_span = timedelta(weeks=6)

    omen_markets = OmenSubgraphHandler().get_omen_markets_simple(
        limit=limit_data,
        filter_by=FilterBy.RESOLVED,
        sort_by=SortBy.NONE,
        include_categorical_markets=False,
        created_after=start_date,
    )
    logger.info(f"Got {len(omen_markets)} Omen in the data.")

    # Plot and save distribution of times
    plt.figure(figsize=(10, 4))
    plt.hist([m.creation_datetime for m in omen_markets], bins=30)
    plt.xlabel("Creation Time")
    plt.ylabel("Count")
    plt.title("Distribution of Omen Market Creation Times")
    plt.tight_layout()
    date_hist_path = os.path.join(output_dir, "creation_datetime_distribution.png")
    plt.savefig(date_hist_path)
    logger.info(f"Saved creation date histogram to {date_hist_path}")

    # Sort markets by creation_datetime ascending
    omen_markets_sorted = sorted(omen_markets, key=lambda m: m.creation_datetime)
    latest_creation_datetime = omen_markets_sorted[-1].creation_datetime

    # Compute split boundaries
    test_cutoff = latest_creation_datetime - test_span
    val_cutoff = test_cutoff - val_span

    # Assign splits
    train_markets = [m for m in omen_markets_sorted if m.creation_datetime < val_cutoff]
    val_markets = [
        m
        for m in omen_markets_sorted
        if val_cutoff <= m.creation_datetime < test_cutoff
    ]
    test_markets = [
        m for m in omen_markets_sorted if m.creation_datetime >= test_cutoff
    ]

    logger.info(
        f"Split: {len(train_markets)} train, {len(val_markets)} val, {len(test_markets)} test "
        f"(val_cutoff={val_cutoff}, test_cutoff={test_cutoff})"
    )

    X_train = [m.question.title for m in train_markets]
    y_train = [int(m.question.boolean_outcome) for m in train_markets]
    X_val = [m.question.title for m in val_markets]
    y_val = [int(m.question.boolean_outcome) for m in val_markets]
    X_test = [m.question.title for m in test_markets]
    y_test = [int(m.question.boolean_outcome) for m in test_markets]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(
        f"Label proportions: train={label_prop(y_train)}, val={label_prop(y_val)}, test={label_prop(y_test)}"
    )

    # Oversample minority class in training set
    # Find class counts
    train_counts = Counter(y_train)
    classes = list(train_counts.keys())
    max_count = max(train_counts.values())
    # For each class, get indices
    class_indices = {c: [i for i, y in enumerate(y_train) if y == c] for c in classes}
    # Oversample: for each class, sample with replacement to max_count
    oversampled_indices = []
    rng = np.random.default_rng(seed)
    for c in classes:
        idxs = class_indices[c]
        if len(idxs) < max_count:
            # Sample with replacement
            sampled = rng.choice(idxs, size=max_count, replace=True)
            oversampled_indices.extend(sampled.tolist())
        else:
            oversampled_indices.extend(idxs)
    X_train_modified = [X_train[i] for i in oversampled_indices]
    y_train_modified = [y_train[i] for i in oversampled_indices]
    logger.info(
        f"Oversampled training set to {len(X_train_modified)} (each class count: {max_count})"
    )

    # 3. Model
    total_steps = (len(X_train_modified) // batch_size + 1) * max_epochs
    warmup_steps = int(0.1 * total_steps)
    model = MarketClassifier(
        pretrained_model_name="distilbert-base-uncased",
        lr=1e-4,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        max_length=42,
        dropout_prob=0.01,
        logit_scale=1.0,
        temperature=1.0,
    ).to(device)

    # Plot distribution of true lengths from the training dataset
    train_tokenizer = model.tokenizer
    train_lengths = [
        len(train_tokenizer.encode(q, truncation=False, add_special_tokens=True))
        for q in X_train_modified
    ]
    plt.figure(figsize=(8, 6))
    plt.hist(train_lengths, bins=30, color="orange", edgecolor="black")
    plt.title("Distribution of Tokenized Question Lengths (Train Set)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    length_hist_path = os.path.join(output_dir, "train_question_length_hist.png")
    plt.savefig(length_hist_path)
    plt.close()
    logger.info(f"Saved training question length histogram to {length_hist_path}")

    # 4. Datasets and loaders
    train_dataset = TextDataset(
        X_train_modified, y_train_modified, model.tokenizer, model.max_length
    )
    val_dataset = TextDataset(X_val, y_val, model.tokenizer, model.max_length)
    test_dataset = TextDataset(X_test, y_test, model.tokenizer, model.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 6. Trainer
    logger_tb = TensorBoardLogger(output_dir)
    watched_metric, watched_mode = "val_profits", "max"
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=watched_metric,
        patience=10,
        mode=watched_mode,
        verbose=True,
        min_delta=0.0001,
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=watched_metric,
        mode=watched_mode,
        save_top_k=1,
        dirpath=output_dir,
        filename="best-checkpoint",
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger_tb,
        accelerator=device.type,
        log_every_n_steps=10,
        deterministic=True,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 7. Train
    trainer.fit(model, train_loader, val_loader)

    logger.info(
        f"Loading best model from checkpoint: {checkpoint_callback.best_model_path}"
    )
    model = MarketClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path
    ).to(device)

    # --- Optimize threshold on validation set to maximize profits ---
    logger.info("Optimizing threshold on validation set to maximize profits...")

    # Get predicted probabilities and true labels for validation set
    model.eval()
    val_probs = []
    val_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, "Val predictions"):
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.label.cpu().numpy()
            probs = model(input_ids, attention_mask).cpu().numpy()
            val_probs.extend(probs.tolist())
            val_labels.extend(labels.tolist())

    val_probs = np.array(val_probs)
    val_labels = np.array(val_labels)

    # --- Calibration: Fit Platt scaling or Isotonic Regression on validation set ---
    logger.info(f"Calibrating probabilities using method: {calibration_method}")
    if calibration_method == "platt":
        # Platt scaling: logistic regression on probs
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(val_probs.reshape(-1, 1), val_labels)

        # For sklearn's LogisticRegression, use predict_proba for calibrated probs
        class PlattCalibrator:
            def __init__(self, lr):
                self.lr = lr

            def transform(self, x):
                # x: shape (n, 1)
                return self.lr.predict_proba(x)[:, 1]

        calibrator_obj = PlattCalibrator(calibrator)
    elif calibration_method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_probs, val_labels)

        class IsoCalibrator:
            def __init__(self, iso):
                self.iso = iso

            def transform(self, x):
                # x: shape (n, 1)
                return self.iso.transform(x.flatten())

        calibrator_obj = IsoCalibrator(calibrator)
    else:
        calibrator_obj = None
        logger.warning("No calibration performed, unknown method.")

    # Plot calibration curve before and after calibration
    prob_true, prob_pred = calibration_curve(val_labels, val_probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Uncalibrated")
    if calibrator_obj is not None:
        val_probs_cal = calibrator_obj.transform(val_probs.reshape(-1, 1))
        prob_true_cal, prob_pred_cal = calibration_curve(
            val_labels, val_probs_cal, n_bins=10
        )
        plt.plot(prob_pred_cal, prob_true_cal, marker="o", label="Calibrated")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve (Validation Set)")
    plt.legend()
    cal_curve_path = os.path.join(output_dir, "calibration_curve_val.png")
    plt.savefig(cal_curve_path)
    plt.close()
    logger.info(f"Saved calibration curve to {cal_curve_path}")

    # --- Optimize threshold on calibrated probabilities ---
    thresholds = np.linspace(0.0, 1.0, 100)
    best_profit = -float("inf")
    best_threshold = 0.5
    profits_by_threshold = []

    if calibrator_obj is not None:
        val_probs_for_thresh = calibrator_obj.transform(val_probs.reshape(-1, 1))
    else:
        val_probs_for_thresh = val_probs

    for t in tqdm(thresholds, "Find best threshold"):
        # Simulate profit for this threshold
        pred_targets = (val_probs_for_thresh > t).astype(int)
        # Use the same profit calculation as in ApproxAverageBinaryMarketProfitability
        market_p_yes = 0.5
        max_bet = 1.0
        profit = 0.0
        for pred, target in zip(pred_targets, val_labels):
            if pred == target:
                profit += max_bet / market_p_yes - max_bet
            else:
                profit -= max_bet
        profits_by_threshold.append(profit)
        if profit > best_profit:
            best_profit = profit
            best_threshold = t

    logger.info(
        f"Best threshold on validation set: {best_threshold:.3f} (profit={best_profit:.2f})"
    )

    # Plot profit vs threshold and save
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, profits_by_threshold, marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Total Profit (Validation Set)")
    plt.title("Profit vs Threshold (Validation Set, Calibrated)")
    plt.grid(True)
    profit_vs_thresh_path = os.path.join(output_dir, "profit_vs_threshold_val.png")
    plt.savefig(profit_vs_thresh_path)
    plt.close()
    logger.info(f"Saved profit vs threshold plot to {profit_vs_thresh_path}")

    # Reload the model with best threshold and calibrator configured.
    model = MarketClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        threshold=best_threshold,
        calibrator=calibrator_obj,
    ).to(device)

    # 8. Test
    for set_name, (X, y, loader) in [
        ("Train", (X_train_modified, y_train_modified, train_loader)),
        ("Val", (X_val, y_val, val_loader)),
        ("Test", (X_test, y_test, test_loader)),
    ]:
        logger.info(f"{set_name} metrics:")
        trainer.test(model, loader)
        logger.info(f"Potential profit {len(X) * 1.0}.")

        pred_probs = [model.predict_text_p_yes(q) for q in X]
        pred_labels = [int(p > model.threshold) for p in pred_probs]

        pred_labels_prob = {
            label: pred_labels.count(label) / len(pred_labels)
            for label in set(pred_labels)
        }
        logger.info(f"Predicted label proportions for {set_name}: {pred_labels_prob}")

        # Save histogram of predicted probabilities
        plt.figure(figsize=(8, 6))
        plt.hist(
            pred_probs,
            bins=42,
            color="skyblue",
            edgecolor="black",
            range=(0.0, 1.0),
        )
        plt.title(f"Histogram of Predicted Probabilities ({set_name} Set, Calibrated)")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.xlim(0.0, 1.0)
        plt.grid(axis="y", alpha=0.75)
        hist_path = os.path.join(output_dir, f"{set_name}_pred_probs_hist.png")
        plt.savefig(hist_path)
        plt.close()

    # Eval fixed model so we have base comparison
    for class_ in [0, 1]:
        model.fixed_class = class_
        logger.info(f"Testing metrics on fixed class {class_}:")
        trainer.test(model, test_loader)
        model.fixed_class = None

    if typer.confirm("Do you want to upload the model's best checkpoint to IPFS?"):
        ipfs_hash = upload_to_ipfs(checkpoint_callback.best_model_path)
        logger.info(f"Model's IPFS hash: {ipfs_hash}")


if __name__ == "__main__":
    APP()
