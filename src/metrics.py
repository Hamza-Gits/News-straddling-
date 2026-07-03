"""Performance metrics over a list of TradeResult."""

import numpy as np
import pandas as pd


def trades_to_frame(trades) -> pd.DataFrame:
    rows = [t.__dict__ for t in trades]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['filled'] = df['side'] != 0
    return df


def summarize(trades) -> dict:
    df = trades_to_frame(trades)
    if df.empty or not df['filled'].any():
        return {'n_events': len(df), 'n_trades': 0, 'net_pnl': 0.0,
                'profit_factor': 0.0, 'expectancy': 0.0, 'win_rate': 0.0,
                'max_drawdown': 0.0, 'whipsaw_rate': 0.0, 'fill_rate': 0.0,
                'avg_win': 0.0, 'avg_loss': 0.0, 'sharpe_per_trade': 0.0}
    f = df[df['filled']].copy()
    pnl = f['pnl_dollars'].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = float(np.max(peak - equity)) if len(equity) else 0.0
    gross_w = float(wins.sum()) if len(wins) else 0.0
    gross_l = float(-losses.sum()) if len(losses) else 0.0
    return {
        'n_events': int(len(df)),
        'n_trades': int(len(f)),
        'fill_rate': float(len(f) / len(df)),
        'net_pnl': float(pnl.sum()),
        'profit_factor': float(gross_w / gross_l) if gross_l > 0 else float('inf'),
        'expectancy': float(pnl.mean()),
        'win_rate': float((pnl > 0).mean()),
        'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'max_drawdown': dd,
        'whipsaw_rate': float(f['whipsaw'].mean()),
        'sharpe_per_trade': float(pnl.mean() / pnl.std()) if pnl.std() > 0 else 0.0,
    }
