import hashlib
import os
import sqlite3
from typing import Dict, List
from urllib.request import urlretrieve

import boto3
from botocore import UNSIGNED
from botocore.client import Config
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import quandl
from plotly import colors
from plotly.subplots import make_subplots
import requests
from scipy import stats
from tqdm.notebook import tqdm, trange


# =============================================================================
# Credentials
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Company Universe
# =============================================================================

# Fundamentals
# ------------


def get_table(dataset_code: str, database_code: str = "ZACKS"):
    """Downloads Zacks fundamental table from export api to local zip file."""

    url = (
        f"https://www.quandl.com/api/v3/datatables/{database_code}/{dataset_code}.json"
    )
    r = requests.get(
        url, params={"api_key": os.getenv("QUANDL_API_KEY"), "qopts.export": "true"}
    )
    data = r.json()
    urlretrieve(
        data["datatable_bulk_download"]["file"]["link"],
        f"zacks_{dataset_code.lower()}.zip",
    )


def load_table_files(table_filenames: Dict):
    """Loads Zacks fundamentals tables from csv files."""

    dfs = []
    for v in tqdm(table_filenames.values()):
        dfs.append(pd.read_csv(v, low_memory=False))

    return dfs


def filter_mktv(
    df_mktv: pd.DataFrame,
    df_mt_selected: pd.DataFrame,
    mktv_min: int,
    per_end_date_beg: str,
    per_end_date_end: str,
) -> pd.DataFrame:
    """Filters companies from MKTV table to include only those selected included in the
    filtered MT table with continuous trading history and minimum market value as
    of the beginning of the period.
    """

    npers = len(pd.date_range(per_end_date_beg, per_end_date_end, freq="Q"))

    df_mktv_selected = df_mktv[
        (df_mktv.mkt_val > mktv_min)
        & (df_mktv.per_end_date == per_end_date_beg)
        & (df_mktv.active_ticker_flag == "Y")
        & df_mktv.ticker.isin(df_mt_selected.ticker)
    ]

    continuous = (
        df_mktv[
            df_mktv.ticker.isin(df_mktv_selected.ticker)
            & (df_mktv.per_end_date >= per_end_date_beg)
            & (df_mktv.per_end_date <= per_end_date_end)
        ]
        .groupby("ticker")
        .count()["per_end_date"]
        == npers
    )

    df_mktv_selected = df_mktv[
        df_mktv.ticker.isin(continuous[continuous].index)
        & (df_mktv.per_end_date >= per_end_date_beg)
        & (df_mktv.per_end_date <= per_end_date_end)
    ]

    return df_mktv_selected


def filter_fc(
    df_mktv_selected: pd.DataFrame,
    df_fr: pd.DataFrame,
    df_fc: pd.DataFrame,
    debt_cap_min: float,
) -> pd.DataFrame:
    """Filters FC table to quaterly records based on filtered MKTV table and
    there being at least one period in which `FR/lterm_debt_cap` was greater
    than the minimum specified.
    """

    per_end_date_beg = df_mktv_selected.per_end_date.min()
    per_end_date_end = df_mktv_selected.per_end_date.max()

    df_fr_selected = df_fr[
        df_fr.ticker.isin(df_mktv_selected.ticker.unique())
        & (df_fr.per_end_date >= per_end_date_beg)
        & (df_fr.per_end_date <= per_end_date_end)
    ]

    has_debt_cap = df_fr_selected.groupby("ticker").max().lterm_debt_cap > debt_cap_min
    has_debt_cap = has_debt_cap[has_debt_cap].index

    df_fr_selected = df_fr_selected[df_fr_selected.ticker.isin(has_debt_cap)]

    df_fc_selected = df_fc[
        df_fc.ticker.isin(df_fr_selected.ticker.unique())
        & (df_fc.per_end_date >= per_end_date_beg)
        & (df_fc.per_end_date <= per_end_date_end)
        & (df_fc.filing_type == "10-Q")
    ]

    return df_fc_selected.set_index(["ticker", "per_end_date"]).sort_index()


def filter_missing(df_fc_selected: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Filters FC table to include only those data reported for each of the selected columns."""

    df_fc_selected = df_fc_selected[columns]

    missing_data = df_fc_selected.isna().sum(axis=1).groupby("ticker").sum()
    without_shares = (
        (df_fc_selected.wavg_shares_out_diluted <= 0).groupby("ticker").sum()
    )
    multiple_tickers = (
        df_fc_selected.index.get_level_values("ticker").unique().str.find(".") != -1
    )

    not_missing_data = missing_data[
        (missing_data == 0) & (without_shares == 0) & (multiple_tickers == 0)
    ].index

    return df_fc_selected[
        df_fc_selected.index.get_level_values("ticker").isin(not_missing_data)
    ]


# Prices
# ------


def get_hash(string: str) -> str:
    """Returns md5 hash of string."""

    return hashlib.md5(str(string).encode()).hexdigest()


def fetch_ticker(
    dataset_code: str, query_params: Dict = None, database_code: str = "EOD"
):
    """Fetches price data for a single ticker."""

    url = f"https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}.json"

    params = dict(api_key=os.getenv("QUANDL_API_KEY"))
    if query_params is not None:
        params = dict(**params, **query_params)

    r = requests.get(url, params=params)

    dataset = r.json()["dataset"]
    df = pd.DataFrame(
        dataset["data"], columns=[c.lower() for c in dataset["column_names"]]
    )
    df["ticker"] = dataset["dataset_code"]

    return df.sort_values("date")


def fetch_all_tickers(tickers: List, query_params: Dict) -> pd.DataFrame:
    """Fetches price data from Quandl for each ticker in provide list and
    returns a dataframe of them concatenated together.
    """

    df_prices = pd.DataFrame()
    for t in tqdm(tickers):
        try:
            df = fetch_ticker(t, query_params)
            df_prices = pd.concat([df_prices, df])
        except:
            print(f"Couldn't get prices for {t}.")

    not_missing_data = (
        df_prices.set_index(["ticker", "date"])[["adj_close"]]
        .unstack("date")
        .isna()
        .sum(axis=1)
        == 0
    )

    df_prices = df_prices[
        df_prices.ticker.isin(not_missing_data[not_missing_data].index)
    ]

    return df_prices.set_index(["ticker", "date"])


# =============================================================================
# Download Preprocessed Files from S3
# =============================================================================


def upload_s3_file(filename: str):
    """Uploads file to S3. Requires credentials with write permissions to exist
    as environment variables.
    """

    client = boto3.client("s3")
    client.upload_file(filename, "finm33150", filename)


def download_s3_file(filename: str):
    """Downloads file from read only S3 bucket."""

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    client.download_file("finm33150", filename, filename)


# =============================================================================
# Load Database Tables
# =============================================================================

conn = sqlite3.connect(":memory:")
cursor = conn.cursor()


def load_db_table(df: pd.DataFrame, tablename: str, chunksize: int = 10000):
    """Loads dataframe to sqlite3 database. Expects no index."""

    cursor.execute(f"DROP TABLE IF EXISTS {tablename};").fetchone()

    chunk_size = 10000
    total = len(df)
    n_chunks = total // chunk_size + 1
    for i in trange(n_chunks):
        df.iloc[i * chunk_size : (i + 1) * chunk_size].to_sql(
            tablename, conn, method="multi", if_exists="append", index=False
        )


# =============================================================================
# Evaluation Metrics
# =============================================================================


def get_info_ratio(returns: pd.Series, benchmark: pd.Series) -> float:

    r_rb = returns - benchmark
    mean = r_rb.mean()
    var = r_rb.var()
    return mean / var


def get_ff_returns(filename: str, freq: str = "BM"):
    """Returns dataframe of Fama-French returns grouped by frequency per
    pandas frequency codes with returns aggregated by sum.
    """

    df_ff = pd.read_csv(filename, header=2, index_col=0)
    df_ff.index = pd.to_datetime(
        df_ff.index, infer_datetime_format=False, format="%Y%m%d"
    )
    df_ff.index.name = "date"

    return df_ff.groupby(pd.Grouper(freq=freq)).sum() / 100


# =============================================================================
# Strategy
# =============================================================================


def merge_data(
    strategy_dates: List[str],
    table_name: str = "merged_data",
):
    """Merges price and fundamentals data. Fundamentals data is on a trailing
    four quarters basis for income and cash flow items (if any) and on a most
    recent quarter basis for balance sheet items.
    """

    conn.executescript(
        f"""
        -- Gather trailing four fundamentals rows for each price date
        DROP TABLE IF EXISTS temp_trailing;

        CREATE TABLE temp_trailing AS        
        SELECT * FROM (
            SELECT p.*, f.*, rank() OVER w rank
            FROM prices p
            JOIN fundamentals f ON (
                p.ticker = f.ticker
                AND f.filing_date < p.date
            )
            WHERE p.date IN ({(",").join(strategy_dates)})
            WINDOW w AS (PARTITION BY p.ticker, p.date ORDER BY f.per_end_date DESC)
        ) WHERE rank <= 4;

        
        -- Calc ltm for flows and merge with most recent balance items
        DROP TABLE IF EXISTS {table_name};

        CREATE TABLE {table_name} AS        
        SELECT tt.ticker, tt.date, adj_close,
            wavg_shares_out_diluted, net_lterm_debt,
            oper_income_ltm,
            CASE
                WHEN eps_diluted_cont_oper_ltm < 0
                THEN  0.001
                ELSE eps_diluted_cont_oper_ltm
                END eps_diluted_cont_oper_ltm,
            income_cont_oper_ltm
        FROM temp_trailing tt
        JOIN (
            SELECT ticker, date, SUM(oper_income) oper_income_ltm, 
                SUM(eps_diluted_cont_oper) eps_diluted_cont_oper_ltm,
                SUM(income_cont_oper) income_cont_oper_ltm
            FROM temp_trailing
            GROUP BY ticker, date
        ) ltm ON tt.ticker = ltm.ticker AND tt.date = ltm.date
        WHERE tt.rank = 1;
    """
    ).fetchone()


def get_ntickers(table_name: str = "merged_data"):
    ntickers = cursor.execute(
        f"""SELECT COUNT(DISTINCT ticker) FROM {table_name}"""
    ).fetchone()[0]

    return ntickers


def calc_ratio_single(ratio_sql: str, change: bool = False, table_name: str = "ratios"):

    change_sql = f"""
        UPDATE {table_name}
        SET ratio = (
            SELECT ratio - ratio_prior
                FROM (
                    SELECT rowid, ratio,
                    LAG(ratio) OVER w ratio_prior
                    FROM {table_name}
                    WINDOW w AS (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING
                )
            ) r2 WHERE r2.rowid = {table_name}.rowid
        ) WHERE date > (SELECT MIN(date) FROM {table_name});
    """

    conn.executescript(
        f"""
        -- Copy master table
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT * FROM merged_data;
        
        -- Calculate ratio
        ALTER TABLE {table_name} ADD COLUMN ratio REAL;
        UPDATE {table_name}
        SET ratio = {ratio_sql};

        -- Calculate ratio change
        {change_sql if change else ""}

        -- Delete starting date used for calculating change in ratio
        DELETE FROM {table_name}
        WHERE date = (SELECT MIN(date) FROM {table_name});
        """
    ).fetchone()


def calc_ratio_multiple(ratio_sqls: List[Dict], table_name: str = "ratios"):
    """Calculates statistic to be ranked based on weighted average of four ratios."""

    def get_ratio_rank_sql(
        ratio_label=None, ratio_sql=None, ascending=None, weight=None, change=False
    ):

        change_sql = f"""
            UPDATE {table_name}
            SET ratio_{ratio_label} = (
                SELECT ratio_{ratio_label} - ratio_{ratio_label}_prior
                    FROM (
                        SELECT rowid, ratio_{ratio_label},
                        LAG(ratio_{ratio_label}) OVER w ratio_{ratio_label}_prior
                        FROM {table_name}
                        WINDOW w AS (
                            PARTITION BY ticker ORDER BY date
                            ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING
                    )
                ) r2 WHERE r2.rowid = {table_name}.rowid
            ) WHERE date > (SELECT MIN(date) FROM {table_name});
        """

        desc = "DESC" if not ascending else ""

        return f"""
            ALTER TABLE {table_name} ADD COLUMN ratio_{ratio_label} REAL;
            ALTER TABLE {table_name} ADD COLUMN rank_{ratio_label} INTEGER;

            UPDATE {table_name}
            SET ratio_{ratio_label} = {ratio_sql};

            {change_sql if change else ""}

            UPDATE {table_name}
            SET rank_{ratio_label} = (
                SELECT rank FROM (
                    SELECT rowid, rank() OVER w rank
                    FROM {table_name}
                    WINDOW w AS (PARTITION BY date ORDER BY ratio_{ratio_label} {desc})
                ) r2 WHERE r2.rowid = {table_name}.rowid
            );
            """

    ratio_rank_sql = ("\n").join([get_ratio_rank_sql(**r) for r in ratio_sqls])

    weighted_ratio_sql = [
        f"rank_{r['ratio_label']} * {r['weight']}" for r in ratio_sqls
    ]

    weight_sum = sum([r["weight"] for r in ratio_sqls])

    conn.executescript(
        f"""
        -- Copy master table
        DROP TABLE IF EXISTS {table_name};
        CREATE TABLE {table_name} AS
        SELECT * FROM merged_data;

        -- Calc individual ratio ranks
        {ratio_rank_sql}
        
        -- Calc weighted average rank
        ALTER TABLE {table_name} ADD COLUMN ratio REAL;
        UPDATE {table_name}
        SET ratio = ({("+").join(weighted_ratio_sql)}) / {weight_sum};

        -- Delete starting date used for calculating change in ratio
        DELETE FROM {table_name}
        WHERE date = (SELECT MIN(date) FROM {table_name});
        """
    ).fetchone()


def run_strategy(
    position_amounts: Dict, ascending: bool = True, table_name: str = "ratios"
):
    """Calculates rank from ratio column and then runs strategy, calculating
    share positions and profits at each strategy date.

    Args:
        position_amounts: Dict with rank as key and dollar amount for positions with
            rank equal greater than prior key and less than or equal to current key.
            The keys should be in ascending order.
    """

    desc = "DESC" if not ascending else ""

    start_date, end_date, ntickers = conn.execute(
        f"SELECT MIN(date), MAX(date), COUNT(DISTINCT ticker) FROM {table_name}"
    ).fetchone()

    rank_positions = []
    start_rank = 1
    for k, v in position_amounts.items():
        for r in range(start_rank, k + 1):
            rank_positions.append(f"WHEN {r} THEN {v}")
        start_rank = k + 1

    nrank = max(position_amounts.keys())

    conn.executescript(
        f"""
        ALTER TABLE {table_name} ADD COLUMN ratio_rank INTEGER;
        ALTER TABLE {table_name} ADD COLUMN shares_position REAL;
        ALTER TABLE {table_name} ADD COLUMN profit REAL;

        -- Calculate ratio_rank
        UPDATE {table_name}
        SET ratio_rank = (
            SELECT rank FROM (
                SELECT rowid, rank() OVER w rank
                FROM {table_name}
                WINDOW w AS (PARTITION BY date ORDER BY ratio {desc})
            ) r2 WHERE r2.rowid = {table_name}.rowid
        );
        
        -- Set share_position
        UPDATE {table_name}
        SET shares_position = 0, profit = 0;
        
        UPDATE {table_name}
        SET shares_position =
            CASE ratio_rank {(" ").join(rank_positions)} ELSE 0 END / adj_close
        WHERE ratio_rank <= {nrank}
        AND date < "{end_date}";

        UPDATE {table_name}
        SET shares_position =
            - CASE {ntickers} - ratio_rank + 1 {(" ").join(rank_positions)} ELSE 0 END / adj_close
        WHERE ratio_rank > {ntickers - nrank}
        AND date < "{end_date}";

        -- Calculate profit
        UPDATE {table_name}
        SET profit = (
            SELECT (adj_close - adj_close_prior) * shares_position_prior
                FROM (
                    SELECT rowid, shares_position,
                    LAG(shares_position) OVER w shares_position_prior,
                    LAG(adj_close) OVER w adj_close_prior
                    FROM {table_name}
                    WINDOW w AS (
                        PARTITION BY ticker ORDER BY date
                        ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING
                )
            ) r2 WHERE r2.rowid = {table_name}.rowid
        ) WHERE date > "{start_date}";

    """
    ).fetchone()


def get_strategy_df(
    table_name: str = "ratios", all_columns: bool = False
) -> pd.DataFrame:
    strat_cols = [
        "date",
        "ticker",
        "wavg_shares_out_diluted",
        "adj_close",
        "ratio",
        "ratio_rank",
        "shares_position",
        "profit",
        "adj_close * shares_position dollar_position",
    ]

    all_cols = "*, adj_close * shares_position dollar_position"

    df_strategy = pd.read_sql(
        f"""
        SELECT {all_cols if all_columns else (",").join(strat_cols)}
        FROM {table_name}
    """,
        conn,
    )
    df_strategy.date = pd.to_datetime(df_strategy.date)
    return df_strategy.set_index(["ticker", "date"]).unstack("date")


def get_strat_result_df(
    df_strategy: pd.DataFrame,
    capital_mult: float = 1,
    borrowing_rate: float = 0,
    repo_rate: float = 0,
) -> pd.DataFrame:
    """Profit in the current period relates to shares_position in the prior
    period. Likewise, irrelevant for this exercise, where capital is constant
    throughout, but return in current period is a function of profit in the
    current period and capital as of the end of the prior period.

    For determining the quantile to which profit relates, it is necessary to
    consider the rank as of the end of the prior period, when the share
    position were established, not the rank in current period, which is used
    to determine share positions as of the end of the current period.

    Since we follow a convention of negative share positions for the bottom
    quantile and positive ones for the top, we can attribute profit to the
    correct quantile based on the sign of dollar positions at the end of
    the prior period.
    """

    top_positions = (df_strategy.shares_position > 0).shift(1, axis=1).fillna(False)
    bottom_positions = (df_strategy.shares_position < 0).shift(1, axis=1).fillna(False)

    df_strat_result = pd.DataFrame(
        {
            "top": (df_strategy.profit * top_positions).sum(),
            "bottom": (df_strategy.profit * bottom_positions).sum(),
            "top_dollar_positions": (
                df_strategy.dollar_position.shift(axis=1) * top_positions
            ).sum(),
            "bottom_dollar_positions": (
                -df_strategy.dollar_position.shift(axis=1) * bottom_positions
            ).sum(),
        }
    )

    funding_days = (
        pd.Series(df_strat_result.index, index=df_strat_result.index).diff().dt.days
    )

    total_capital = (
        df_strat_result[["top_dollar_positions", "bottom_dollar_positions"]].sum(axis=1)
        * capital_mult
    )

    funding_cost = (
        df_strat_result.top_dollar_positions * borrowing_rate * funding_days / 360
    ) - (df_strat_result.bottom_dollar_positions * repo_rate * funding_days / 360)

    df_strat_result[f"top_quantile"] = df_strat_result.top.divide(total_capital)
    df_strat_result[f"bottom_quantile"] = df_strat_result.bottom.divide(total_capital)

    df_strat_result["strat_return"] = np.log(
        1 + df_strat_result[["top", "bottom"]].sum(axis=1).divide(total_capital)
    )

    df_strat_result["bm_return"] = np.log(
        df_strategy.adj_close / df_strategy.adj_close.shift(1, axis=1)
    ).mean()
    df_strat_result["ntickers"] = len(df_strategy)

    return df_strat_result.fillna(0)


def check_strat(df_strategy: pd.DataFrame):
    """Quick check to make sure everything is working correctly.
    * Rank is consistent with ratio
    * Profit is calculated correctly for changes in adj_close
        * Increase in adj_close for top ranked results in increase in profit
        * Decrease in adj_close for bottom ranked results in increase in profit
        * Returns are positive when profit is positive
    * Note that returns are expressed in relation to aggregate capital to
        facilitate aggregation by ranking group
    """

    df_strat_check = df_strategy.iloc[
        :, df_strategy.columns.get_level_values("date") > df_strategy.ratio.columns[-5]
    ][
        [
            "ratio",
            "ratio_rank",
            "shares_position",
            "adj_close",
            "profit",
            "dollar_position",
        ]
    ].sort_values(
        ("ratio_rank", df_strategy.ratio.columns[-1])
    )

    return df_strat_check


def get_strat_metrics_single(
    ratio_label: str,
    ratio_sql: str,
    ascending: bool,
    change: bool,
    position_amounts: Dict,
) -> Dict:
    """Gets strategy metrics - total return, standard deviation, information ratio,
    beta and downside beta - for a single strategy.
    """

    params = dict(
        ratio_label=ratio_label,
        ratio_sql=ratio_sql,
        ascending=ascending,
        change=change,
        position_amounts=position_amounts,
    )

    calc_ratio_single(ratio_sql, change=change)
    run_strategy(position_amounts, ascending)

    df_strategy = get_strategy_df(all_columns=False)
    assert not df_strategy.isna().any().any()

    df_strat_result = get_strat_result_df(df_strategy)

    strat_return = df_strat_result["strat_return"].sum()
    strat_std = df_strat_result["strat_return"].std()
    info_ratio = get_info_ratio(df_strat_result.strat_return, df_strat_result.bm_return)
    beta = df_strat_result.strat_return.corr(df_strat_result.bm_return)
    downside_beta = df_strat_result[df_strat_result.bm_return < 0]["strat_return"].corr(
        df_strat_result[df_strat_result.bm_return < 0]["bm_return"]
    )

    metrics = dict(
        strat_return=strat_return,
        strat_std=strat_std,
        beta=beta,
        downside_beta=downside_beta,
        info_ratio=info_ratio,
    )

    metrics = {k: round(v, 4) for k, v in metrics.items()}

    return {**params, **metrics}


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.D3


def get_companies_by_sector_chart(
    df_mt: pd.DataFrame, excluded_sectors: List
) -> go.Figure:

    fig = go.Figure(
        data=[
            go.Bar(
                x=df_mt.zacks_x_sector_desc.value_counts().index,
                y=df_mt.zacks_x_sector_desc.value_counts(),
                marker_color=list(
                    map(
                        lambda x: COLORS[int(x)],
                        df_mt.zacks_x_sector_desc.value_counts().index.isin(
                            excluded_sectors
                        ),
                    )
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=(
            f"Count of Companies by Sector - "
            f"{df_mt.zacks_x_sector_desc.value_counts().sum():,.0f} in Total"
        )
    )

    return fig


def get_moments_annotation(
    s: pd.Series, xref: str, yref: str, x: float, y: float, xanchor: str, title: str
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """

    labels = [
        ("obs", lambda x: f"{x:>16d}"),
        ("min:max", lambda x: f"{x[0]:>0.3f}:{x[1]:>0.3f}"),
        ("mean", lambda x: f"{x:>13.4f}"),
        ("std", lambda x: f"{x:>15.4f}"),
        ("skewness", lambda x: f"{x:>11.4f}"),
        ("kurtosis", lambda x: f"{x:>13.4f}"),
    ]

    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    return go.layout.Annotation(
        text=(
            f"{title}:<br>"
            + ("<br>").join(
                [f"{k[0]:<10}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=1,
        borderpad=2,
        bgcolor="white",
        font=dict(size=10),
        xanchor=xanchor,
        yanchor="top",
    )


def get_strat_chart(
    df_strat_result: pd.DataFrame, ratio_label: str, height=800
) -> go.Figure:

    strat_return = df_strat_result["strat_return"].sum()
    bm_return = df_strat_result["bm_return"].sum()
    strat_color = 0
    bm_color = 1

    fig = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.15,
        subplot_titles=(
            "Returns from Top and Bottom Quantiles<sup>1</sup>",
            "Cumulative Returns vs. Benchmark<sup>2</sup>",
            "Returns vs. Benchmark - Downside",
            "Returns vs. Benchmark",
        ),
    )

    # Strategy Returns
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result.strat_return,
            name=df_strat_result.strat_return.name,
            line=dict(color=COLORS[strat_color]),
            legendgroup=0,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result.bm_return,
            name=df_strat_result.bm_return.name,
            line=dict(color=COLORS[bm_color]),
            legendgroup=1,
        ),
        row=2,
        col=2,
    )

    # Cumulative Returns
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result.strat_return.cumsum(),
            line=dict(color=COLORS[strat_color]),
            showlegend=False,
            name="strat_return",
            legendgroup=0,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result.bm_return.cumsum(),
            line=dict(color=COLORS[bm_color]),
            showlegend=False,
            name="bm_return",
            legendgroup=1,
        ),
        row=1,
        col=2,
    )

    # Downside Returns
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result[df_strat_result.bm_return < 0]["strat_return"],
            line=dict(color=COLORS[strat_color]),
            showlegend=False,
            name="strat_return",
            legendgroup=0,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df_strat_result.index,
            y=df_strat_result[df_strat_result.bm_return < 0]["bm_return"],
            line=dict(color=COLORS[bm_color]),
            showlegend=False,
            name="bm_return",
            legendgroup=1,
        ),
        row=2,
        col=1,
    )

    # Top and bottom quantiles
    fig.add_trace(
        go.Bar(
            x=df_strat_result.index,
            y=df_strat_result.top_quantile,
            offsetgroup=0,
            marker_color=COLORS[2],
            name=df_strat_result.top_quantile.name,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df_strat_result.index,
            y=df_strat_result.bottom_quantile,
            offsetgroup=0,
            marker_color=COLORS[3],
            name=df_strat_result.bottom_quantile.name,
        ),
        row=1,
        col=1,
    )

    fig.add_annotation(
        get_moments_annotation(
            df_strat_result["strat_return"],
            xref="paper",
            yref="paper",
            x=1.03,
            y=0.30,
            xanchor="left",
            title="strategy",
        )
    )

    fig.add_annotation(
        get_moments_annotation(
            df_strat_result["bm_return"],
            xref="paper",
            yref="paper",
            x=1.03,
            y=0.13,
            xanchor="left",
            title="benchmark",
        )
    )

    info_ratio = get_info_ratio(df_strat_result.strat_return, df_strat_result.bm_return)
    beta = df_strat_result.strat_return.corr(df_strat_result.bm_return)
    downside_beta = df_strat_result[df_strat_result.bm_return < 0]["strat_return"].corr(
        df_strat_result[df_strat_result.bm_return < 0]["bm_return"]
    )

    fig.add_annotation(
        text=(
            f"Info Ratio: {info_ratio:>11.4f}<br>"
            f"Beta: {beta:>19.4f}<br>"
            f"Down Beta: {downside_beta:>10.4f}"
        ),
        xref="paper",
        yref="paper",
        x=1.03,
        y=0.35,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        text=(f"<b>{strat_return:0.4f}<b>"),
        xref="paper",
        yref="y2",
        x=1.01,
        y=strat_return,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        text=(f"{bm_return:0.4f}"),
        xref="paper",
        yref="y2",
        x=1.01,
        y=bm_return,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        text=(
            "Notes:"
            "<br><sup>1</sup>Quantile returns based on profit divided by total capital to "
            "approximate strategy return when added together."
            "<br><sup>2</sup>Benchmark returns based on average share price return for "
            f"{df_strat_result.ntickers.max()} companies from which quantiles were selected."
        ),
        xref="paper",
        yref="paper",
        x=-0.05,
        y=-0.10,
        xanchor="left",
        showarrow=False,
        align="left",
        font=dict(size=10),
    )

    fig.update_layout(
        title_text=f"Quantile Strategy Returns: {ratio_label}",
        height=height,
        legend=dict(yanchor="top", y=0.61, xanchor="left", x=1.03),
    )

    return fig


def get_ratio_comp_df(
    df_strategy: pd.DataFrame, nrank: int, ratio_name: str
) -> pd.DataFrame:

    ratio_records = []
    for d in df_strategy.ratio.columns:
        top = df_strategy.ratio[d][df_strategy.ratio_rank[d] <= nrank]
        ratio_records.append(
            {
                "date": d,
                "quantile": "top",
                "ratio_25": top.quantile(0.25),
                "ratio_75": top.quantile(0.75),
                "ratio_med": top.median(),
            }
        )

        bottom = df_strategy.ratio[d][
            df_strategy.ratio_rank[d] > (len(df_strategy.index) - nrank)
        ]
        ratio_records.append(
            {
                "date": d,
                "quantile": "bottom",
                "ratio_25": bottom.quantile(0.25),
                "ratio_75": bottom.quantile(0.75),
                "ratio_med": bottom.median(),
            }
        )

    df = pd.DataFrame(ratio_records).set_index(["date", "quantile"]).unstack("quantile")
    df.name = ratio_name

    return df


def get_ratio_comp_chart(
    df_ratio_comp: pd.DataFrame,
    title_text: str = "Comparison of Quantile Ratios",
    height: int = 400,
) -> go.Figure:

    fig = go.Figure()

    color_map = {"top": 2, "bottom": 3}
    for q in df_ratio_comp.columns.get_level_values(1).unique():
        fig.add_trace(
            go.Scatter(
                x=df_ratio_comp.index,
                y=df_ratio_comp.ratio_med[q],
                name=f"{q}",
                line=dict(color=COLORS[color_map[q]]),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_ratio_comp.index,
                y=df_ratio_comp.ratio_75[q],
                name=f"{q}_75pct",
                line=dict(width=0),
                marker=dict(color=COLORS[color_map[q]]),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df_ratio_comp.index,
                y=df_ratio_comp.ratio_25[q],
                name=f"{q}_25pct",
                line=dict(width=0),
                marker=dict(color=COLORS[color_map[q]]),
                fill="tonexty",
                showlegend=False,
            )
        )

    fig.update_layout(
        title_text=f"{title_text}: {df_ratio_comp.name}",
        height=height,
    )

    fig.add_annotation(
        text=(
            f"Line represents median ratio of quantile. Bands represent 25th and 75th percentile within"
            " strategy quantile."
        ),
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.25,
        xanchor="left",
        align="left",
        showarrow=False,
    )

    return fig


def get_ranking_change_chart(
    df_strategy: pd.DataFrame, ratio_label: str, height=500
) -> go.Figure:
    df_hm = df_strategy.ratio_rank.sort_values(df_strategy.profit.columns[0])
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(x=df_hm.columns, y=df_hm.index, z=df_hm, colorscale="RdYlGn_r")
    )
    fig["layout"]["yaxis"]["autorange"] = "reversed"
    fig.update_layout(title_text="Change in Ranking Over Time")
    fig.add_annotation(
        text=(
            f"Colormap tied to rank of ratio. Tickers are sorted in ascending order at the start of"
            " the strategy<br>and then remain in the same order throughout with color updated to reflect "
            "ratio rank in subsequent periods."
        ),
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.15,
        xanchor="left",
        align="left",
        showarrow=False,
    )

    fig.update_layout(
        title_text=f"Change in Ranking Over Time: {ratio_label}",
        height=height,
    )

    return fig
