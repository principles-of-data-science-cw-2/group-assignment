"""
eda.py — Exploratory Data Analysis Module
==========================================
Module  : Principles of Data Science (7144)
Section : Exploratory Data Analysis
Author  : Anuradha Dhananjanee

This script contains reusable functions used in 02_eda.ipynb.
It can be run standalone or imported as a module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Visual style ─────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams.update({
    'figure.dpi'      : 120,
    'axes.titlesize'  : 13,
    'axes.titleweight': 'bold',
    'axes.labelsize'  : 11,
})


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load cleaned retail CSV and create time-based columns.

    Parameters
    ----------
    data_path : str
        Full path to cleaned_retail_data.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with added Month, Hour, DayOfWeek, Week columns
    """
    df = pd.read_csv(data_path)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Create time-based columns
    df['Month']     = df['InvoiceDate'].dt.month
    df['Hour']      = df['InvoiceDate'].dt.hour
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['Week']      = df['InvoiceDate'].dt.isocalendar().week.astype(int)

    # Detect flexible column names
    df.attrs['customer_col'] = 'Customer ID' if 'Customer ID' in df.columns else 'CustomerID'
    df.attrs['invoice_col']  = 'Invoice'     if 'Invoice'     in df.columns else 'InvoiceNo'

    print(f"Data loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range : {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    return df


# ============================================================
# 2. DESCRIPTIVE STATISTICS
# ============================================================

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for Quantity, Price, Total_Revenue
    including coefficient of variation, skewness, and kurtosis.
    """
    cols      = ['Quantity', 'Price', 'Total_Revenue']
    stats     = df[cols].describe().T
    stats['cv%']      = (stats['std'] / stats['mean'] * 100).round(2)
    stats['skew']     = df[cols].skew().values
    stats['kurtosis'] = df[cols].kurtosis().values
    return stats.round(3)


def dataset_overview(df: pd.DataFrame) -> None:
    """Print a summary overview of the dataset."""
    customer_col = df.attrs.get('customer_col', 'Customer ID')
    invoice_col  = df.attrs.get('invoice_col',  'Invoice')

    print('=== DATASET OVERVIEW ===')
    print(f'  Total transactions  : {len(df):,}')
    print(f'  Unique invoices     : {df[invoice_col].nunique():,}')
    print(f'  Unique customers    : {df[customer_col].nunique():,}')
    print(f'  Unique products     : {df["StockCode"].nunique():,}')
    print(f'  Unique countries    : {df["Country"].nunique()}')
    print(f'  Total Revenue (GBP) : £{df["Total_Revenue"].sum():,.2f}')
    print(f'  Missing values      :\n{df.isnull().sum()}')


# ============================================================
# 3. REVENUE TREND FUNCTIONS
# ============================================================

def monthly_revenue_trend(df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
    """
    Plot monthly revenue trend with 3-month rolling average.
    Returns the monthly revenue DataFrame.
    """
    monthly = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Total_Revenue'].sum().reset_index()
    monthly['InvoiceDate']  = monthly['InvoiceDate'].dt.to_timestamp()
    monthly['Rolling_Avg']  = monthly['Total_Revenue'].rolling(3).mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly['InvoiceDate'], monthly['Total_Revenue'],
            marker='o', color='#2E75B6', linewidth=2.5, markersize=7, label='Monthly Revenue')
    ax.plot(monthly['InvoiceDate'], monthly['Rolling_Avg'],
            color='orange', linestyle='--', linewidth=1.5, label='3-Month Rolling Avg')
    ax.fill_between(monthly['InvoiceDate'], monthly['Total_Revenue'], alpha=0.12, color='#2E75B6')

    peak_idx = monthly['Total_Revenue'].idxmax()
    ax.annotate(f'Peak: £{monthly.loc[peak_idx,"Total_Revenue"]:,.0f}',
                xy=(monthly.loc[peak_idx,'InvoiceDate'], monthly.loc[peak_idx,'Total_Revenue']),
                xytext=(20, -40), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='red'), color='red', fontweight='bold')

    ax.set_title('Monthly Revenue Trend')
    ax.set_xlabel('Month')
    ax.set_ylabel('Total Revenue (£)')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return monthly


def revenue_by_day_and_hour(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot revenue split by day of week and hour of day."""
    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    day_rev   = df.groupby('DayOfWeek')['Total_Revenue'].sum().reindex(day_order)
    hour_rev  = df.groupby('Hour')['Total_Revenue'].sum().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    bar_colors = ['#C0392B' if d == 'Sunday' else '#2E75B6' for d in day_order]
    axes[0].bar(day_rev.index, day_rev.values, color=bar_colors, edgecolor='white')
    axes[0].set_title('Revenue by Day of Week')
    axes[0].set_ylabel('Total Revenue (£)')
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M'))

    peak_hour = hour_rev.idxmax()
    axes[1].bar(hour_rev.index, hour_rev.values, color='#1ABC9C', edgecolor='white')
    axes[1].axvline(peak_hour, color='red', linestyle='--', linewidth=1.5, label=f'Peak: {peak_hour}:00')
    axes[1].set_title('Revenue by Hour of Day')
    axes[1].set_ylabel('Total Revenue (£)')
    axes[1].legend()

    plt.suptitle('Temporal Demand Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Peak day  : {day_rev.idxmax()}')
    print(f'Peak hour : {peak_hour}:00')


# ============================================================
# 4. PRODUCT ANALYSIS FUNCTIONS
# ============================================================

def top_products_by_revenue(df: pd.DataFrame, n: int = 10, save_path: str = None) -> pd.DataFrame:
    """Return and plot top N products by total revenue."""
    top = df.groupby('Description')['Total_Revenue'].sum().nlargest(n).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(top['Description'], top['Total_Revenue'], color='#1ABC9C', edgecolor='white')
    ax.set_title(f'Top {n} Products by Total Revenue')
    ax.set_xlabel('Total Revenue (£)')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x:,.0f}'))
    for bar, val in zip(bars, top['Total_Revenue']):
        ax.text(val + 500, bar.get_y() + bar.get_height()/2, f'£{val:,.0f}', va='center', fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return top


def price_band_analysis(df: pd.DataFrame, save_path: str = None) -> None:
    """Segment products into price bands and plot count vs revenue."""
    bins   = [0, 1, 3, 5, 10, 50]
    labels = ['<£1', '£1-3', '£3-5', '£5-10', '£10+']
    df = df.copy()
    df['Price_Band'] = pd.cut(df['Price'], bins=bins, labels=labels)

    count = df['Price_Band'].value_counts().reindex(labels)
    rev   = df.groupby('Price_Band', observed=True)['Total_Revenue'].sum().reindex(labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(count.index, count.values, color='#9B59B6', edgecolor='white')
    axes[0].set_title('Transaction Count by Price Band')
    axes[0].set_ylabel('Number of Transactions')

    axes[1].bar(rev.index, rev.values, color='#E67E22', edgecolor='white')
    axes[1].set_title('Total Revenue by Price Band')
    axes[1].set_ylabel('Revenue (£)')
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'£{x/1e6:.1f}M'))

    plt.suptitle('Product Price Band Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. CUSTOMER ANALYSIS FUNCTIONS
# ============================================================

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) table.

    Returns
    -------
    pd.DataFrame
        One row per customer with Recency (days), Frequency (orders), Monetary (£)
    """
    customer_col   = df.attrs.get('customer_col', 'Customer ID')
    invoice_col    = df.attrs.get('invoice_col',  'Invoice')
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        Recency   = ('InvoiceDate',   lambda x: (reference_date - x.max()).days),
        Frequency = (invoice_col,     'nunique'),
        Monetary  = ('Total_Revenue', 'sum')
    ).reset_index()
    return rfm


def pareto_analysis(df: pd.DataFrame, save_path: str = None) -> float:
    """
    Plot Pareto curve for revenue concentration by customer.
    Returns the % of customers that generate 80% of revenue.
    """
    customer_col   = df.attrs.get('customer_col', 'Customer ID')
    cust_rev       = df.groupby(customer_col)['Total_Revenue'].sum().sort_values(ascending=False)
    cumulative_pct = cust_rev.cumsum() / cust_rev.sum() * 100
    customer_pct   = np.arange(1, len(cust_rev)+1) / len(cust_rev) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(customer_pct, cumulative_pct.values, color='#2E75B6', linewidth=2)
    ax.axhline(80, color='red',    linestyle='--', linewidth=1.5, label='80% Revenue')
    ax.axvline(20, color='orange', linestyle='--', linewidth=1.5, label='Top 20% Customers')
    ax.fill_between(customer_pct, cumulative_pct.values, alpha=0.1, color='#2E75B6')
    ax.set_title('Pareto Analysis — Revenue Concentration by Customer', fontweight='bold')
    ax.set_xlabel('Cumulative % of Customers')
    ax.set_ylabel('Cumulative % of Revenue')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    idx_80 = np.searchsorted(cumulative_pct.values, 80)
    pct_for_80 = customer_pct[idx_80]
    print(f'{pct_for_80:.1f}% of customers generate 80% of total revenue.')
    return pct_for_80


# ============================================================
# 6. ANOMALY DETECTION
# ============================================================

def detect_anomalies(df: pd.DataFrame, column: str = 'Total_Revenue',
                     n_std: float = 3.0) -> tuple:
    """
    Detect anomalies using mean + n*std threshold.

    Returns
    -------
    tuple : (anomalies DataFrame, threshold float)
    """
    threshold = df[column].mean() + n_std * df[column].std()
    anomalies = df[df[column] > threshold]
    print(f'Threshold (mean + {n_std}σ): £{threshold:.2f}')
    print(f'Anomalies found: {len(anomalies):,} ({len(anomalies)/len(df)*100:.2f}%)')
    return anomalies, threshold


# ============================================================
# 7. MAIN — RUN FULL EDA PIPELINE
# ============================================================

if __name__ == '__main__':
    # Resolve paths relative to this script's location
    BASE_FOLDER    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_FILE      = os.path.join(BASE_FOLDER, 'data', 'cleaned_retail_data.csv')
    REPORTS_FOLDER = os.path.join(BASE_FOLDER, 'reports')
    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    # Load data
    df = load_data(DATA_FILE)

    # Overview
    dataset_overview(df)

    # Descriptive stats
    print('\n=== DESCRIPTIVE STATISTICS ===')
    print(descriptive_stats(df).to_string())

    # Revenue trends
    monthly_revenue_trend(df, save_path=os.path.join(REPORTS_FOLDER, 'eda_monthly_revenue.png'))
    revenue_by_day_and_hour(df, save_path=os.path.join(REPORTS_FOLDER, 'eda_temporal_patterns.png'))

    # Products
    top_products_by_revenue(df, n=10, save_path=os.path.join(REPORTS_FOLDER, 'eda_top_products.png'))
    price_band_analysis(df, save_path=os.path.join(REPORTS_FOLDER, 'eda_price_bands.png'))

    # Customers
    rfm = compute_rfm(df)
    print('\n=== RFM SUMMARY ===')
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2))
    pareto_analysis(df, save_path=os.path.join(REPORTS_FOLDER, 'eda_pareto.png'))

    # Anomalies
    anomalies, threshold = detect_anomalies(df)

    print('\nEDA pipeline complete. Charts saved to reports/ folder.')
