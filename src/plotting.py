import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no GUI needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PLOTS_DIR = "plots"


def _save(fig: plt.Figure, filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def plot_price_signals(df: pd.DataFrame) -> None:
    """Price + Bollinger Bands + Buy/Sell markers."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df["close"],    color="#4C9BE8", lw=1.2, label="Close")
    ax.plot(df.index, df["BB_MA"],    color="#A0A0A0", lw=0.8, ls="--", label="BB Mid")
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                    alpha=0.12, color="#A0A0A0", label="BB Band")

    buys  = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    ax.scatter(buys.index,  buys["close"],  marker="^", color="#2ECC71", s=60, zorder=5, label="Buy")
    ax.scatter(sells.index, sells["close"], marker="v", color="#E74C3C", s=60, zorder=5, label="Sell")

    ax.set_title("Price · Bollinger Bands · Signals", fontsize=13, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    _save(fig, "price_signals.png")


def plot_rsi(df: pd.DataFrame) -> None:
    """RSI with overbought/oversold bands."""
    fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(df.index, df["RSI"], color="#9B59B6", lw=1.2)
    ax.axhline(70, color="#E74C3C", lw=0.8, ls="--", label="Overbought (70)")
    ax.axhline(30, color="#2ECC71", lw=0.8, ls="--", label="Oversold (30)")
    ax.fill_between(df.index, 70, df["RSI"].clip(lower=70), alpha=0.15, color="#E74C3C")
    ax.fill_between(df.index, df["RSI"].clip(upper=30), 30, alpha=0.15, color="#2ECC71")

    ax.set_title("RSI (14)", fontsize=13, fontweight="bold")
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    _save(fig, "rsi.png")


def plot_adx(df: pd.DataFrame) -> None:
    """ADX with trend-strength threshold."""
    fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(df.index, df["ADX"], color="#E67E22", lw=1.2, label="ADX")
    ax.axhline(25, color="#888", lw=0.8, ls="--", label="Trend threshold (25)")
    ax.fill_between(df.index, 25, df["ADX"].clip(lower=25), alpha=0.15, color="#E67E22")

    ax.set_title("ADX (14) — Trend Strength", fontsize=13, fontweight="bold")
    ax.set_ylabel("ADX")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    _save(fig, "adx.png")


def plot_portfolio(portfolio: pd.DataFrame, initial_capital: float) -> None:
    """Portfolio value over time vs. buy-and-hold baseline."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(portfolio.index, portfolio["value"], color="#4C9BE8", lw=1.4, label="Strategy")
    ax.axhline(initial_capital, color="#888", lw=0.8, ls="--", label=f"Capital (${initial_capital:,.0f})")

    ax.set_title("Portfolio Value Over Time", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    _save(fig, "portfolio.png")


def plot_dashboard(df: pd.DataFrame, portfolio: pd.DataFrame,
                   initial_capital: float) -> None:
    """Unified 4-panel dashboard with a shared time axis."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5, 2],
                                          "hspace": 0.08})
    fig.suptitle("Trading Strategy Dashboard — AAPL",
                 fontsize=16, fontweight="bold", y=0.95)

    # ── Panel 1: Price + Bollinger Bands + Signals ────────────────────── #
    ax = axes[0]
    ax.plot(df.index, df["close"], color="#4C9BE8", lw=1.2, label="Close")
    ax.plot(df.index, df["BB_MA"], color="#A0A0A0", lw=0.8, ls="--", label="BB Mid")
    ax.fill_between(df.index, df["BB_lower"], df["BB_upper"],
                    alpha=0.12, color="#A0A0A0", label="BB Band")

    buys  = df[df["signal"] == 1]
    sells = df[df["signal"] == -1]
    ax.scatter(buys.index,  buys["close"],  marker="^", color="#2ECC71",
               s=80, zorder=5, label="Buy")
    ax.scatter(sells.index, sells["close"], marker="v", color="#E74C3C",
               s=80, zorder=5, label="Sell")

    ax.set_ylabel("Price (USD)", fontsize=10)
    ax.set_title("Price · Bollinger Bands · Signals", fontsize=12,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    ax.grid(alpha=0.25)

    # ── Panel 2: RSI ──────────────────────────────────────────────────── #
    ax = axes[1]
    ax.plot(df.index, df["RSI"], color="#9B59B6", lw=1.2)
    ax.axhline(70, color="#E74C3C", lw=0.8, ls="--", label="Overbought (70)")
    ax.axhline(30, color="#2ECC71", lw=0.8, ls="--", label="Oversold (30)")
    ax.fill_between(df.index, 70, df["RSI"].clip(lower=70),
                    alpha=0.15, color="#E74C3C")
    ax.fill_between(df.index, df["RSI"].clip(upper=30), 30,
                    alpha=0.15, color="#2ECC71")
    ax.set_ylabel("RSI", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("RSI (14)", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)

    # ── Panel 3: ADX ─────────────────────────────────────────────────── #
    ax = axes[2]
    ax.plot(df.index, df["ADX"], color="#E67E22", lw=1.2, label="ADX")
    ax.axhline(25, color="#888", lw=0.8, ls="--", label="Trend threshold (25)")
    ax.fill_between(df.index, 25, df["ADX"].clip(lower=25),
                    alpha=0.15, color="#E67E22")
    ax.set_ylabel("ADX", fontsize=10)
    ax.set_title("ADX (14) — Trend Strength", fontsize=12,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(alpha=0.25)

    # ── Panel 4: Portfolio Value ──────────────────────────────────────── #
    ax = axes[3]
    ax.plot(portfolio.index, portfolio["value"], color="#4C9BE8", lw=1.4,
            label="Strategy")
    ax.axhline(initial_capital, color="#888", lw=0.8, ls="--",
               label=f"Capital (${initial_capital:,.0f})")
    ax.fill_between(portfolio.index, initial_capital, portfolio["value"],
                    where=portfolio["value"] >= initial_capital,
                    alpha=0.10, color="#2ECC71", label="Profit zone")
    ax.fill_between(portfolio.index, initial_capital, portfolio["value"],
                    where=portfolio["value"] < initial_capital,
                    alpha=0.10, color="#E74C3C", label="Loss zone")
    ax.set_ylabel("Value (USD)", fontsize=10)
    ax.set_title("Portfolio Value Over Time", fontsize=12,
                 fontweight="bold", loc="left")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(alpha=0.25)

    # Shared x-axis formatting (only on bottom panel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate(rotation=30)

    _save(fig, "dashboard.png")


def plot_all(df: pd.DataFrame, portfolio: pd.DataFrame, initial_capital: float) -> None:
    """Render the merged dashboard."""
    print("\n=== Generating Plots ===")
    plot_dashboard(df, portfolio, initial_capital)
