#region imports
from AlgorithmImports import *
#endregion

"""
This file contains configuration, costants and stock names.
"""


DATA_SAVE_DIR = "datasets"
TRAINED_MODEL_DIR = "trained_models"
TENSORBOARD_LOG_DIR = "tensorboard_log"
RESULTS_DIR = "results"


# TODO: Mod the below code to allow using multiple csv files on my terms
CSV_FILE_SETTINGS = {
    "baseline_file_name": "/home/gich2023/Future~WSL/Lanuvo Research/WIO/DRL4Trading-main/Mod/tickers_data/dow30/^DJI.csv",
    "dir_list": ["/home/gich2023/Future~WSL/Lanuvo Research/WIO/DRL4Trading-main/Mod/tickers_data/dow30"],

    "has_daily_trading_limit": False,

    "use_baseline_data": True,
    
    "date_column_name": "datadate",
    "baseline_date_column_name": "Date"
}

CSV_FIELD_MAPPINGS = {
    "datadate": "date",
    "prcod": "open",
    "prchd": "high",
    "prcld": "low",
    "prccd": "close",
    "cshtrd": "volume",
    "ajexdi": "adj_factor",
}

BASELINE_FIELD_MAPPINGS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
}


IN_DIR = 'tickers_data'
CSV_DIR = 'csv_data'
EXP_FILE_NAME = 'combined_csv.csv'


# self defined
################ CfD Ticker Setup Start###################
# TODO Finish setting up the CfDs that are available
# Separate those that I'll will use at the begining
BaseEquityCfDs = [
    "SPX500USD", "NAS100USD", "US30USD", "US2000USD", "UK100GBP", "GER30EUR", "JPN225JPY", "FRA40EUR", "HKG33HKD", "CHN50RMB", "SUI30EUR", "US10YBond", 
]

BaseCommodityCfDs = [
    "SPX500USD", "NAS100USD", "US30USD", "US2000USD", "EUR", "BBRI.JK",
    "BBTN.JK", "BMRI.JK", "BSDE.JK", "INDF.JK", "JPFA.JK", "JSMR.JK",
    "KLBF.JK", "PGAS.JK", "PJAA.JK", "PPRO.JK", "SIDO.JK", "SMGR.JK",
    "TINS.JK", "TLKM.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSKT.JK",
    "WTON.JK"
]
################ CfD Ticker Setup End###################

################ FX Ticker Setup Start###################
FX_TICKER = [
    "AUDCAD", "AUDCHF", "AUDJPY", "AUDNZD", "AUDSGD", "AUDUSD", "AUDUSD",
    "AUDUSD", "AUDUSD", "AUDUSD", "AUDUSD", "AUDUSD", "CADCHF", "CADHKD",
    "CADJPY", "CHFJPY", "CHFSGD", "EURAUD", "EURCAD", "EURCHF", 
    "EURCHF",
    "EURCHF", "EURCZK", "EURGBP", "EURHKD", "EURHUF", "EURJPY", "EURNOK",
    "EURNZD", "EURPLN", "EURRUB", "EURSEK", "EURSGD", "EURTRY", "EURTRY",
    "EURUSD", "GBPAUD", "GBPAUD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPJPY",
    "GBPNZD", "GBPUSD", "HKDJPY", "NZDCAD", "NZDCHF", "NZDJPY", "NZDUSD",
    "SGDJPY", "TRYJPY", "USDCAD", "USDCHF", "USDCNH", "USDCZK", "USDHKD",
    "USDHUF", "USDILS", "USDJPY", "USDMXN", "USDNOK", "USDPLN", "USDRON",
    "USDRUB", "USDSEK", "USDSGD", "USDTHB", "USDTRY", "USDZAR", "XAGUSD",
    "XAUUSD", "ZARJPY", "EURDKK"
    ]
################FX Ticker Setup End###################