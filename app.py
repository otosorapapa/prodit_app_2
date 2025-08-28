# -*- coding: utf-8 -*-
"""
EC収益管理アプリ（楽天中小事業者向け） — Streamlit単一ファイル実装
要件対応：
- 0. 目的/KGI/KPI：ダッシュボード&集計応答高速化（st.cache_data）
- 1. 利用者/権限：アプリ内シンプル認証 + ロール（管理者/経営者/担当者/監査）
- 2. 機能要件 FR-001〜FR-007（MUST）+ FR-101〜105（SHOULD）+ 拡張一部
- 3. 画面要件：ダッシュボード/データ取込/在庫管理/利益分析/返品・不良/RFM/設定/監査ログ
- 4. データ要件：SQLite+SQLAlchemy（型&整合性チェック）、CSVインポート/エクスポート
- 5. 外部IF：Slack Webhook通知、PDF出力（reportlab/日本語フォント）
- 6. NFR：キャッシュ最適化、バックアップ（ZIP）、監視ログ
- 7. 設計：Streamlit内完結（将来FastAPI分離可能な構造）

★ セットアップ（ローカル）
pip install streamlit pandas numpy SQLAlchemy requests reportlab python-dateutil pytz openpyxl
# （任意）SQLCipher 暗号化利用時：pip install pysqlcipher3
# 実行：streamlit run app.py

★ 初期ログイン（st.secrets未設定時のデフォルト）
- admin / admin （管理者）
- owner / owner （経営者）
- staff / staff （担当者）
- audit / audit （監査）

★ secrets.toml 例（.streamlit/secrets.toml）
[auth]
admin = "pbkdf2:sha256:demo"  # デモ：プレーン照合も可
owner = "owner"
staff = "staff"
audit = "audit"
[roles]
admin = "admin"
owner = "executive"
staff = "staff"
audit = "auditor"
[slack]
webhook_url = "https://hooks.slack.com/services/xxx/yyy/zzz"
[db]
path = "data/app.sqlite3"  # SQLCipher使用時は "data/app.db" をPRAGMA keyで暗号化
[app]
company_name = "くらしいきいき株式会社"
lead_time_days = 14
safety_stock_default = 5

"""
import os
import io
import re
import json
import zipfile
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
import requests
from dateutil.relativedelta import relativedelta

import streamlit as st
from sqlalchemy import (
    create_engine, Column, Integer, String, Date, DateTime, Float, Numeric, Boolean, Text, UniqueConstraint,
    ForeignKey, event
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import IntegrityError

# =============== 基本設定 ===============
APP_TZ = pytz.timezone("Asia/Tokyo")
TODAY = datetime.now(APP_TZ).date()
DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
REPORT_DIR = os.path.join(DATA_DIR, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =============== DB 接続 ===============
Base = declarative_base()

def get_db_path() -> str:
    try:
        db_path = st.secrets["db"].get("path", os.path.join(DATA_DIR, "app.sqlite3"))
    except Exception:
        db_path = os.path.join(DATA_DIR, "app.sqlite3")
    return db_path

@st.cache_resource(show_spinner=False)
def get_engine():
    db_path = get_db_path()
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, future=True)
    return engine

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())

# =============== スキーマ定義 ===============
class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    order_id = Column(String, unique=True, index=True, nullable=False)
    order_date = Column(Date, index=True, nullable=False)
    channel = Column(String, index=True, default="Rakuten")
    buyer_id = Column(String, index=True)
    subtotal = Column(Numeric(14, 2), default=0)
    shipping_fee = Column(Numeric(14, 2), default=0)
    coupon_discount = Column(Numeric(14, 2), default=0)
    points_used = Column(Numeric(14, 2), default=0)
    platform_fee = Column(Numeric(14, 2), default=0)
    tax = Column(Numeric(14, 2), default=0)
    status = Column(String, default="paid")
    created_at = Column(DateTime, default=lambda: datetime.now(APP_TZ))

    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")

class OrderItem(Base):
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(String, ForeignKey("orders.order_id", ondelete="CASCADE"), index=True)
    sku = Column(String, index=True)
    qty = Column(Integer, default=0)
    unit_price = Column(Numeric(14, 2), default=0)
    cogs = Column(Numeric(14, 2), default=0)
    shipping_alloc = Column(Numeric(14, 2), default=0)
    fee_alloc = Column(Numeric(14, 2), default=0)
    ad_cost_alloc = Column(Numeric(14, 2), default=0)

    order = relationship("Order", back_populates="items")

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    sku = Column(String, unique=True, index=True)
    product_name = Column(String)
    category = Column(String, index=True)
    cost = Column(Numeric(14, 2), default=0)
    price = Column(Numeric(14, 2), default=0)
    supplier = Column(String)
    safety_stock = Column(Integer, default=5)

class Inventory(Base):
    __tablename__ = "inventory"
    id = Column(Integer, primary_key=True)
    sku = Column(String, index=True)
    warehouse = Column(String, default="MAIN")
    qty_onhand = Column(Integer, default=0)
    qty_allocated = Column(Integer, default=0)
    updated_at = Column(DateTime, default=lambda: datetime.now(APP_TZ))
    __table_args__ = (UniqueConstraint("sku", "warehouse", name="uq_inv_sku_wh"),)

class Return(Base):
    __tablename__ = "returns"
    id = Column(Integer, primary_key=True)
    order_id = Column(String, index=True)
    sku = Column(String, index=True)
    qty = Column(Integer, default=0)
    reason = Column(String)
    defect_flag = Column(Boolean, default=False)
    restocked_flag = Column(Boolean, default=False)
    return_date = Column(Date, default=lambda: datetime.now(APP_TZ).date())

class AdCost(Base):
    __tablename__ = "ad_costs"
    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True)
    campaign = Column(String, index=True)
    channel = Column(String, index=True, default="Rakuten")
    cost = Column(Numeric(14, 2), default=0)
    clicks = Column(Integer, default=0)
    impressions = Column(Integer, default=0)

class Fee(Base):
    __tablename__ = "fees"
    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True)
    channel = Column(String, index=True, default="Rakuten")
    fee_type = Column(String)
    amount = Column(Numeric(14, 2), default=0)
    rule_id = Column(String)

class Customer(Base):
    __tablename__ = "customers"
    id = Column(Integer, primary_key=True)
    buyer_id = Column(String, unique=True, index=True)
    first_order = Column(Date)
    last_order = Column(Date)
    lifetime_value = Column(Numeric(14, 2), default=0)
    r = Column(Integer, default=0)
    f = Column(Integer, default=0)
    m = Column(Integer, default=0)

class Setting(Base):
    __tablename__ = "settings"
    id = Column(Integer, primary_key=True)
    key = Column(String, unique=True)
    value = Column(Text)  # JSON文字列

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True)
    ts = Column(DateTime, default=lambda: datetime.now(APP_TZ))
    actor = Column(String)
    action = Column(String)
    detail = Column(Text)

class Snooze(Base):
    __tablename__ = "snoozes"
    id = Column(Integer, primary_key=True)
    sku = Column(String, index=True)
    until_ts = Column(DateTime)

# =============== 初期化 ===============

def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)
    # 最低限の設定の初期化
    with SessionLocal() as db:
        if not db.query(Setting).filter_by(key="company_name").first():
            cname = st.secrets.get("app", {}).get("company_name", "自社EC")
            db.add(Setting(key="company_name", value=json.dumps(cname, ensure_ascii=False)))
        if not db.query(Setting).filter_by(key="lead_time_days").first():
            lead = st.secrets.get("app", {}).get("lead_time_days", 14)
            db.add(Setting(key="lead_time_days", value=json.dumps(int(lead))))
        if not db.query(Setting).filter_by(key="safety_stock_default").first():
            ss = st.secrets.get("app", {}).get("safety_stock_default", 5)
            db.add(Setting(key="safety_stock_default", value=json.dumps(int(ss))))
        db.commit()

# =============== ユーティリティ ===============

def log(action: str, detail: str = ""):
    actor = st.session_state.get("user", "guest")
    with SessionLocal() as db:
        db.add(AuditLog(actor=actor, action=action, detail=detail[:2000]))
        db.commit()

@st.cache_data(show_spinner=False)
def load_table_df(table_name: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql_table(table_name, engine)
    return df

@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes, encoding: str = "utf-8") -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)

@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)

@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# =============== 認証/権限 ===============
ROLES = {
    "admin": ["dashboard", "import", "inventory", "profit", "returns", "rfm", "settings", "audit"],
    "executive": ["dashboard", "profit", "inventory", "rfm"],
    "staff": ["dashboard", "import", "inventory", "returns", "profit"],
    "auditor": ["audit"],
}

DEFAULT_USERS = {
    "admin": {"password": "admin", "role": "admin"},
    "owner": {"password": "owner", "role": "executive"},
    "staff": {"password": "staff", "role": "staff"},
    "audit": {"password": "audit", "role": "auditor"},
}

def resolve_users_from_secrets() -> Dict[str, Dict[str, str]]:
    users = {}
    try:
        auth = st.secrets.get("auth", {})
        roles = st.secrets.get("roles", {})
        for uname, pwd in auth.items():
            role = roles.get(uname, "staff")
            users[uname] = {"password": str(pwd), "role": role}
    except Exception:
        pass
    if not users:
        users = DEFAULT_USERS
    return users


def login_panel():
    users = resolve_users_from_secrets()
    st.sidebar.markdown("### ログイン")
    uname = st.sidebar.text_input("ユーザー名", value=st.session_state.get("user", ""))
    pwd = st.sidebar.text_input("パスワード", type="password")
    login = st.sidebar.button("ログイン")
    if login:
        if uname in users and str(pwd) == str(users[uname]["password"]):
            st.session_state["user"] = uname
            st.session_state["role"] = users[uname]["role"]
            st.sidebar.success(f"{uname} としてログイン。ロール: {st.session_state['role']}")
            log("login", f"user={uname}")
        else:
            st.sidebar.error("認証失敗。ユーザー名/パスワードをご確認ください。")

    if st.session_state.get("user"):
        if st.sidebar.button("ログアウト"):
            log("logout", st.session_state.get("user"))
            for k in ["user", "role"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()


def role_allows(page_key: str) -> bool:
    role = st.session_state.get("role")
    if not role:
        return False
    return page_key in ROLES.get(role, [])

# =============== Slack 通知 ===============

def send_slack(text: str):
    url = st.secrets.get("slack", {}).get("webhook_url")
    if not url:
        return False, "webhook未設定"
    try:
        r = requests.post(url, json={"text": text}, timeout=10)
        return r.status_code == 200, r.text
    except Exception as e:
        return False, str(e)

# =============== データ入出力 ===============
REQUIRED_MAP = {
    "orders": ["order_id", "order_date", "channel", "buyer_id", "subtotal", "shipping_fee", "coupon_discount", "points_used", "platform_fee", "tax", "status"],
    "order_items": ["order_id", "sku", "qty", "unit_price", "cogs", "shipping_alloc", "fee_alloc", "ad_cost_alloc"],
    "products": ["sku", "product_name", "category", "cost", "price", "supplier", "safety_stock"],
    "inventory": ["sku", "warehouse", "qty_onhand", "qty_allocated", "updated_at"],
    "returns": ["order_id", "sku", "qty", "reason", "defect_flag", "restocked_flag", "return_date"],
    "ad_costs": ["date", "campaign", "channel", "cost", "clicks", "impressions"],
    "fees": ["date", "channel", "fee_type", "amount", "rule_id"],
    "customers": ["buyer_id", "first_order", "last_order", "lifetime_value"],
}

TABLE_CLASS = {
    "orders": Order,
    "order_items": OrderItem,
    "products": Product,
    "inventory": Inventory,
    "returns": Return,
    "ad_costs": AdCost,
    "fees": Fee,
    "customers": Customer,
}


def save_mapping(name: str, mapping: Dict[str, str]):
    with SessionLocal() as db:
        db.merge(Setting(key=f"mapping:{name}", value=json.dumps(mapping, ensure_ascii=False)))
        db.commit()


def load_mapping(name: str) -> Dict[str, str]:
    with SessionLocal() as db:
        row = db.query(Setting).filter_by(key=f"mapping:{name}").first()
        if row:
            return json.loads(row.value)
        return {}


def normalize_df(df: pd.DataFrame, mapping: Dict[str, str], required_cols: List[str]) -> pd.DataFrame:
    # mapping: {app_col -> source_col}
    out = pd.DataFrame()
    for col in required_cols:
        src = mapping.get(col)
        if src in df.columns:
            out[col] = df[src]
        else:
            out[col] = np.nan
    # 型整形
    if "order_date" in out.columns:
        out["order_date"] = pd.to_datetime(out["order_date"], errors="coerce").dt.date
    if "return_date" in out.columns:
        out["return_date"] = pd.to_datetime(out["return_date"], errors="coerce").dt.date
    if "updated_at" in out.columns:
        out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
    # 数値系
    num_cols = [c for c in out.columns if c not in ["order_id", "order_date", "channel", "buyer_id", "sku", "product_name", "category", "supplier", "status", "fee_type", "campaign", "rule_id", "warehouse", "reason", "return_date", "updated_at"]]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    # 文字
    str_cols = ["order_id", "channel", "buyer_id", "sku", "product_name", "category", "supplier", "status", "fee_type", "campaign", "rule_id", "warehouse", "reason"]
    for c in str_cols:
        if c in out.columns:
            out[c] = out[c].fillna("").astype(str)
    return out


def upsert_dataframe(df: pd.DataFrame, table: str, pk: Optional[str] = None):
    engine = get_engine()
    cls = TABLE_CLASS[table]
    with SessionLocal() as db:
        if table == "orders" and pk == "order_id":
            # 重複防止：未存在の注文のみ追加
            exist_ids = set([r[0] for r in db.query(Order.order_id).filter(Order.order_id.in_(df["order_id"].unique().tolist())).all()])
            insert_df = df[~df["order_id"].isin(exist_ids)].copy()
            if not insert_df.empty:
                insert_df.to_sql(table, engine, if_exists="append", index=False, method="multi")
            added = len(insert_df)
            skipped = len(df) - added
            log("import_orders", f"added={added}, skipped={skipped}")
            return added, skipped
        else:
            df.to_sql(table, engine, if_exists="append", index=False, method="multi")
            log("import_table", f"table={table}, rows={len(df)}")
            return len(df), 0

# =============== 指標計算 ===============

@st.cache_data(show_spinner=False)
def get_profit_frame(date_from: date, date_to: date, channels: List[str], categories: List[str]) -> pd.DataFrame:
    eng = get_engine()
    orders = pd.read_sql_query(
        "SELECT * FROM orders WHERE order_date BETWEEN :f AND :t",
        eng,
        params={"f": date_from, "t": date_to},
        parse_dates=["order_date"],
    )
    items = pd.read_sql_table("order_items", eng)
    products = pd.read_sql_table("products", eng)
    ad = pd.read_sql_query(
        "SELECT date, channel, SUM(cost) AS ad_cost FROM ad_costs WHERE date BETWEEN :f AND :t GROUP BY date, channel",
        eng,
        params={"f": date_from, "t": date_to},
        parse_dates=["date"],
    )
    if channels:
        orders = orders[orders["channel"].isin(channels)]
        ad = ad[ad["channel"].isin(channels)] if not ad.empty else ad
    df = items.merge(orders, on="order_id", how="inner", suffixes=("_i", "_o"))
    df = df.merge(products[["sku", "category"]], on="sku", how="left")
    if categories:
        df = df[df["category"].isin(categories)]

    # 注文レベル値引（coupon/points）を明細按分
    rev_by_order = df.groupby("order_id")["unit_price"].sum().rename("order_rev")
    df = df.join(rev_by_order, on="order_id")
    for col in ["coupon_discount", "points_used", "shipping_fee", "platform_fee", "tax", "subtotal"]:
        if col not in df.columns:
            df[col] = 0
    df["rev"] = df["unit_price"].astype(float) * df["qty"].astype(float)
    share = (df["rev"] / df["order_rev"]).replace([np.inf, -np.inf], 0).fillna(0)
    df["coupon_alloc"] = share * df["coupon_discount"].astype(float)
    df["points_alloc"] = share * df["points_used"].astype(float)
    df["ship_alloc_order"] = share * df["shipping_fee"].astype(float)
    df["platfee_alloc_order"] = share * df["platform_fee"].astype(float)

    # 収益分解
    for c in ["cogs", "shipping_alloc", "fee_alloc", "ad_cost_alloc"]:
        if c not in df.columns:
            df[c] = 0
    df["ad_alloc_total"] = df["ad_cost_alloc"].astype(float)  # itemレベルの事前按分
    df["fee_total"] = df["fee_alloc"].astype(float) + df["platfee_alloc_order"].astype(float)
    df["ship_total"] = df["shipping_alloc"].astype(float) + df["ship_alloc_order"].astype(float)
    df["discount_total"] = df["coupon_alloc"].astype(float) + df["points_alloc"].astype(float)

    df["gross_profit"] = df["rev"].astype(float) - (
        df["cogs"].astype(float) + df["ship_total"] + df["fee_total"] + df["ad_alloc_total"] - df["discount_total"]
    )
    df["profit_margin"] = np.where(df["rev"]>0, df["gross_profit"] / df["rev"], np.nan)

    return df

@st.cache_data(show_spinner=False)
def kpi_summary(df: pd.DataFrame) -> Dict[str, float]:
    sales = float(df["rev"].sum())
    gp = float(df["gross_profit"].sum())
    pm = (gp / sales) if sales > 0 else 0
    ad = float(df.get("ad_alloc_total", pd.Series([0]*len(df))).sum())
    return {"sales": sales, "gp": gp, "pm": pm, "ad": ad}

@st.cache_data(show_spinner=False)
def abc_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    s = df.groupby("sku")["gross_profit"].sum().sort_values(ascending=False)
    cum = s.cumsum() / s.sum() if s.sum() != 0 else s.cumsum()
    out = pd.DataFrame({"gp": s, "cum_share": cum})
    out["class"] = pd.cut(out["cum_share"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"], include_lowest=True)
    out = out.reset_index()
    return out

@st.cache_data(show_spinner=False)
def returns_rate(date_from: date, date_to: date) -> float:
    eng = get_engine()
    sold = pd.read_sql_query(
        "SELECT SUM(qty) AS qty FROM order_items oi JOIN orders o ON oi.order_id=o.order_id WHERE o.order_date BETWEEN :f AND :t",
        eng, params={"f": date_from, "t": date_to},
    )["qty"].fillna(0).iloc[0]
    ret = pd.read_sql_query(
        "SELECT SUM(qty) AS qty FROM returns WHERE return_date BETWEEN :f AND :t",
        eng, params={"f": date_from, "t": date_to}
    )["qty"].fillna(0).iloc[0]
    return float(ret) / float(sold) if sold else 0.0

# =============== 需要予測 v1（移動平均/単回帰） ===============
@st.cache_data(show_spinner=False)
def forecast_sku_monthly(sales_df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """sales_df: columns=[order_date, sku, qty] 月次集計後に予測。"""
    if sales_df.empty:
        return pd.DataFrame(columns=["sku", "yhat"])
    g = sales_df.copy()
    g["ym"] = pd.to_datetime(g["order_date"]).dt.to_period("M").dt.to_timestamp()
    m = g.groupby(["sku", "ym"])["qty"].sum().reset_index()
    out_rows = []
    for sku, sub in m.groupby("sku"):
        sub = sub.sort_values("ym")
        # 移動平均
        ma = sub["qty"].rolling(3, min_periods=1).mean()
        # 単回帰（時間→数量）
        x = np.arange(len(sub))
        if len(sub) >= 2:
            coef = np.polyfit(x, sub["qty"].values, 1)
            trend_next = np.polyval(coef, len(sub))
        else:
            trend_next = sub["qty"].iloc[-1]
        yhat = float((ma.iloc[-1] + trend_next) / 2)
        out_rows.append({"sku": sku, "yhat": max(yhat, 0)})
    return pd.DataFrame(out_rows)

# =============== RFM 分析 ===============
@st.cache_data(show_spinner=False)
def rfm_scores(date_ref: date) -> pd.DataFrame:
    eng = get_engine()
    orders = pd.read_sql_table("orders", eng, parse_dates=["order_date"])  
    items = pd.read_sql_table("order_items", eng)
    df = items.merge(orders, on="order_id", how="inner")
    df["amount"] = df["unit_price"].astype(float) * df["qty"].astype(float)
    agg = df.groupby("buyer_id").agg(
        last_order=("order_date", "max"),
        freq=("order_id", "nunique"),
        monetary=("amount", "sum"),
    ).reset_index()
    agg["recency_days"] = (pd.to_datetime(date_ref) - agg["last_order"]).dt.days
    # 五分位スコア（Rは小さいほど良い→逆順）
    agg["R"] = pd.qcut(agg["recency_days"].rank(method="first", ascending=True), 5, labels=[5,4,3,2,1]).astype(int)
    agg["F"] = pd.qcut(agg["freq"].rank(method="first", ascending=False), 5, labels=[5,4,3,2,1]).astype(int)
    agg["M"] = pd.qcut(agg["monetary"].rank(method="first", ascending=False), 5, labels=[5,4,3,2,1]).astype(int)
    agg["RFM"] = agg["R"].astype(str) + agg["F"].astype(str) + agg["M"].astype(str)
    return agg

# =============== 在庫/アラート ===============
@st.cache_data(show_spinner=False)
def inventory_alerts() -> pd.DataFrame:
    eng = get_engine()
    inv = pd.read_sql_table("inventory", eng)
    prod = pd.read_sql_table("products", eng)
    df = inv.merge(prod[["sku", "product_name", "category", "safety_stock"]], on="sku", how="left")
    df["safety_stock"].fillna(load_int("safety_stock_default", 5), inplace=True)
    df["alert"] = df["qty_onhand"] < df["safety_stock"]
    return df[df["alert"]]

# =============== 設定ロード/保存 ===============

def save_setting(key: str, value):
    with SessionLocal() as db:
        db.merge(Setting(key=key, value=json.dumps(value, ensure_ascii=False)))
        db.commit()

@st.cache_data(show_spinner=False)
def load_setting(key: str, default=None):
    with SessionLocal() as db:
        row = db.query(Setting).filter_by(key=key).first()
        if row:
            return json.loads(row.value)
        return default

def load_int(key: str, default: int) -> int:
    v = load_setting(key, default)
    try:
        return int(v)
    except Exception:
        return default

# =============== PDF 出力 ===============

def make_dashboard_pdf(summary: Dict[str, float], abc_table: pd.DataFrame) -> bytes:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))

    # 日本語フォント（HeiseiMin-W3）
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
        font_name = "HeiseiMin-W3"
    except Exception:
        font_name = "Helvetica"

    c.setFont(font_name, 16)
    c.drawString(20*mm, 190*mm, f"ダッシュボードサマリー（{datetime.now(APP_TZ).strftime('%Y-%m-%d')}）")

    c.setFont(font_name, 12)
    y = 175*mm
    def row(lbl, val):
        nonlocal y
        c.drawString(20*mm, y, f"{lbl}")
        c.drawRightString(260*mm, y, f"{val:,.0f}")
        y -= 8*mm

    row("売上", summary.get("sales", 0))
    row("粗利", summary.get("gp", 0))
    row("利益率(%)", summary.get("pm", 0) * 100)
    row("広告費", summary.get("ad", 0))

    # ABC上位
    c.drawString(20*mm, y, "ABC上位（上位10件）")
    y -= 8*mm
    top = abc_table.head(10)
    for _, r in top.iterrows():
        c.drawString(25*mm, y, f"{r['sku']}")
        c.drawRightString(200*mm, y, f"GP: {r['gp']:,.0f}")
        c.drawRightString(260*mm, y, f"累計: {r['cum_share']*100:,.1f}% / {r['class']}")
        y -= 6*mm

    c.showPage()
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# =============== バックアップ ===============

def make_backup_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        db_path = get_db_path()
        if os.path.exists(db_path):
            z.write(db_path, arcname=os.path.basename(db_path))
        for root, _, files in os.walk(UPLOAD_DIR):
            for f in files:
                full = os.path.join(root, f)
                arc = os.path.relpath(full, DATA_DIR)
                z.write(full, arc)
    return buf.getvalue()

# =============== UI 構築 ===============

def sidebar_filters():
    st.sidebar.markdown("### 共通フィルタ")
    today = TODAY
    start = st.sidebar.date_input("開始日", value=today - relativedelta(months=1))
    end = st.sidebar.date_input("終了日", value=today)
    eng = get_engine()
    try:
        channels = pd.read_sql_query("SELECT DISTINCT channel FROM orders", eng)["channel"].dropna().tolist()
    except Exception:
        channels = []
    ch_sel = st.sidebar.multiselect("チャネル", options=channels, default=channels)
    try:
        categories = pd.read_sql_query("SELECT DISTINCT category FROM products", eng)["category"].dropna().tolist()
    except Exception:
        categories = []
    cat_sel = st.sidebar.multiselect("カテゴリ", options=categories)
    return start, end, ch_sel, cat_sel


def page_dashboard():
    st.header("ダッシュボード")
    start, end, ch, cat = sidebar_filters()
    with st.spinner("集計中..."):
        df = get_profit_frame(start, end, ch, cat)
    summ = kpi_summary(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("売上", f"{summ['sales']:,.0f}")
    c2.metric("粗利", f"{summ['gp']:,.0f}")
    c3.metric("利益率", f"{summ['pm']*100:,.1f}%")
    c4.metric("広告費", f"{summ['ad']:,.0f}")
    rr = returns_rate(start, end)
    c5.metric("返品率", f"{rr*100:,.2f}%")

    st.markdown("#### 推移")
    if not df.empty:
        ts = df.groupby(pd.to_datetime(df["order_date"]).dt.date).agg(sales=("rev", "sum"), gp=("gross_profit", "sum"))
        st.line_chart(ts)
        by_ch = df.groupby("channel")["rev"].sum().sort_values(ascending=False)
        st.bar_chart(by_ch)
    else:
        st.info("対象期間のデータがありません。")

    st.markdown("#### SKU別貢献（Pareto）/ 在庫アラート / 返品率Top")
    colA, colB = st.columns([2,1])
    abc = abc_by_sku(df) if not df.empty else pd.DataFrame()
    with colA:
        if not abc.empty:
            st.dataframe(abc.head(100))
            csv = df_to_csv_bytes(abc)
            st.download_button("ABC上位CSV", csv, file_name="abc_top.csv", mime="text/csv")
    with colB:
        alerts = inventory_alerts()
        if not alerts.empty:
            st.dataframe(alerts[["sku","product_name","qty_onhand","safety_stock","warehouse"]])
            if st.button("Slack通知（在庫アラートTop送信）"):
                top = alerts.head(10).copy()
                text = "在庫アラート\n" + "\n".join([f"{r.sku} {r.product_name} 残:{r.qty_onhand}/閾:{r.safety_stock}" for _, r in top.iterrows()])
                ok, resp = send_slack(text)
                st.success("送信しました" if ok else f"失敗: {resp}")

    # PDFエクスポート
    if st.button("PDFレポート出力（A4横）"):
        pdf = make_dashboard_pdf(summ, abc)
        st.download_button("ダウンロード: dashboard.pdf", data=pdf, file_name="dashboard.pdf", mime="application/pdf")


def page_import():
    st.header("データ取込（CSV/Excel）")
    st.caption("複数ファイル一括取込、マッピング保存可。重複注文はスキップ。")

    tab1, tab2 = st.tabs(["アップロード", "マッピング管理"])

    with tab1:
        target_table = st.selectbox("取込先テーブル", list(REQUIRED_MAP.keys()), index=0)
        mapping = load_mapping(target_table)
        uploaded = st.file_uploader("CSV/Excelを選択（複数可）", type=["csv","xlsx","xls"], accept_multiple_files=True)
        if uploaded:
            for uf in uploaded:
                st.write(f"**{uf.name}**")
                content = uf.read()
                # 保存
                with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as fp:
                    fp.write(content)
                # 読み込み
                if uf.type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel") or uf.name.endswith((".xlsx",".xls")):
                    df = read_excel_cached(content)
                    if isinstance(df, dict):
                        # 最初のシート
                        df = list(df.values())[0]
                else:
                    df = read_csv_cached(content)
                st.write("プレビュー（先頭100行）")
                st.dataframe(df.head(100))

                # マッピングUI
                required = REQUIRED_MAP[target_table]
                st.markdown("##### 項目マッピング")
                mcols = {}
                for col in required:
                    src = st.selectbox(f"{col}", options=[""] + list(df.columns), index=( [""] + list(df.columns) ).index(mapping.get(col, "")) if mapping.get(col, "") in df.columns else 0, key=f"map_{uf.name}_{col}")
                    if src:
                        mcols[col] = src
                if st.button(f"このファイルを取込（{uf.name}）", key=f"import_{uf.name}"):
                    norm = normalize_df(df, mcols, required)
                    added, skipped = upsert_dataframe(norm, target_table, pk="order_id" if target_table=="orders" else None)
                    st.success(f"取込成功：追加 {added} 件 / スキップ {skipped} 件")
                    log("import", f"table={target_table}, file={uf.name}, add={added}, skip={skipped}")

        st.divider()
        if st.button("フルバックアップ（DB+アップロード）ZIP作成"):
            z = make_backup_zip()
            st.download_button("ダウンロード: backup.zip", data=z, file_name="backup.zip", mime="application/zip")

    with tab2:
        st.markdown("#### 取込マッピング保存/読込")
        target_table2 = st.selectbox("対象テーブル", list(REQUIRED_MAP.keys()), index=0, key="map_table")
        current = load_mapping(target_table2)
        st.json(current)
        if st.button("現在のセッション内マッピング（上のUIで最後に選んだもの）を保存", help="UIの選択状態はファイルごとにキー付けされます。保存はテーブル単位で集約して保存します。"):
            # セッションキーから回収
            required = REQUIRED_MAP[target_table2]
            mcols = {}
            for col in required:
                # 最後のアップロードファイル名がキーに載るため、直近の値を拾う
                for k in list(st.session_state.keys())[::-1]:
                    if k.startswith("map_") and k.endswith(col):
                        v = st.session_state[k]
                        if v:
                            mcols[col] = v
                            break
            if mcols:
                save_mapping(target_table2, mcols)
                st.success("保存しました")
            else:
                st.info("保存対象のマッピングが見つかりません。上タブで一度選択してください。")


def page_inventory():
    st.header("在庫管理")
    # 一覧
    eng = get_engine()
    inv = pd.read_sql_table("inventory", eng)
    prod = pd.read_sql_table("products", eng)
    df = inv.merge(prod[["sku","product_name","category","safety_stock"]], on="sku", how="left")
    st.dataframe(df)

    st.markdown("#### 閾値設定とアラート")
    default_ss = load_int("safety_stock_default", 5)
    new_default = st.number_input("デフォルト安全在庫", min_value=0, value=default_ss)
    if st.button("保存（デフォルト安全在庫）"):
        save_setting("safety_stock_default", int(new_default))
        st.success("保存しました")

    alerts = inventory_alerts()
    st.markdown("##### アラート一覧")
    if alerts.empty:
        st.success("在庫アラートはありません。")
    else:
        st.dataframe(alerts)
        if st.button("Slack通知（全件）"):
            text = "在庫アラート（全件）\n" + "\n".join([f"{r.sku} {r.product_name} 残:{r.qty_onhand}/閾:{r.safety_stock}" for _, r in alerts.iterrows()])
            ok, resp = send_slack(text)
            st.success("送信しました" if ok else f"失敗: {resp}")


def page_profit():
    st.header("利益分析（収益分解ピボット）")
    start, end, ch, cat = sidebar_filters()
    with st.spinner("ピボット集計..."):
        df = get_profit_frame(start, end, ch, cat)
    if df.empty:
        st.info("データがありません。")
        return

    dims = st.multiselect("次元（列）", ["order_date","channel","category","sku"], default=["order_date","channel"])
    metrics = st.multiselect("指標", ["rev","gross_profit","profit_margin","cogs","ship_total","fee_total","ad_alloc_total","discount_total","qty"], default=["rev","gross_profit","profit_margin"])

    tmp = df.copy()
    tmp["qty"] = tmp["qty"].astype(float)
    pivot = tmp.pivot_table(index=dims, values=[m for m in metrics if m in tmp.columns], aggfunc={m: np.sum for m in metrics if m != "profit_margin"}, observed=False)
    if "profit_margin" in metrics:
        gp = tmp.groupby(dims)["gross_profit"].sum()
        rv = tmp.groupby(dims)["rev"].sum()
        pm = (gp/rv).replace([np.inf, -np.inf], np.nan)
        pivot["profit_margin"] = pm
    pivot = pivot.fillna(0).reset_index()
    st.dataframe(pivot.head(1000))

    st.download_button("CSVエクスポート", df_to_csv_bytes(pivot), file_name="profit_pivot.csv")


def page_returns():
    st.header("返品・不良管理")
    eng = get_engine()
    df = pd.read_sql_table("returns", eng)
    st.dataframe(df.tail(500))

    st.markdown("#### 返品登録（在庫へ同時反映）")
    with st.form("ret_form"):
        order_id = st.text_input("注文ID")
        sku = st.text_input("SKU")
        qty = st.number_input("数量", min_value=1, value=1)
        reason = st.text_input("理由タグ", value="不良")
        defect = st.checkbox("不良フラグ", value=True)
        restock = st.checkbox("在庫戻し", value=True)
        rdate = st.date_input("返品日", value=TODAY)
        sub = st.form_submit_button("登録")
    if sub:
        with SessionLocal() as db:
            db.add(Return(order_id=order_id, sku=sku, qty=int(qty), reason=reason, defect_flag=defect, restocked_flag=restock, return_date=rdate))
            if restock:
                inv = db.query(Inventory).filter_by(sku=sku, warehouse="MAIN").first()
                if not inv:
                    inv = Inventory(sku=sku, warehouse="MAIN", qty_onhand=0, qty_allocated=0)
                    db.add(inv)
                inv.qty_onhand += int(qty)
                inv.updated_at = datetime.now(APP_TZ)
            db.commit()
        st.success("登録しました")
        log("return_reg", f"order={order_id}, sku={sku}, qty={qty}")
        st.experimental_rerun()


def page_rfm():
    st.header("RFM分析 & 顧客抽出")
    ref = st.date_input("基準日", value=TODAY)
    df = rfm_scores(ref)
    st.dataframe(df)

    # 上位セグメント抽出（例：R>=4, F>=4, M>=4）
    st.markdown("#### セグメント抽出")
    rmin = st.slider("R最小", 1, 5, 4)
    fmin = st.slider("F最小", 1, 5, 4)
    mmin = st.slider("M最小", 1, 5, 4)
    seg = df[(df.R>=rmin)&(df.F>=fmin)&(df.M>=mmin)].copy()
    st.dataframe(seg)
    st.download_button("CSV出力（上位セグ）", df_to_csv_bytes(seg), file_name="rfm_top.csv")


def page_settings():
    st.header("設定 / プラグイン化 / 通知ルール")
    st.subheader("Slack/帳票/DB")
    st.text_input("Slack Webhook（secrets.toml 推奨）", value=st.secrets.get("slack", {}).get("webhook_url", ""), disabled=True, help=".streamlit/secrets.toml に設定してください")
    lead = st.number_input("発注リードタイム（日）", min_value=0, value=load_int("lead_time_days", 14))
    if st.button("保存（リードタイム）"):
        save_setting("lead_time_days", int(lead))
        st.success("保存しました")

    st.subheader("通知ルール（粗利/在庫/返品）")
    rules_json = load_setting("notify_rules", {"gp_drop_pct": 20, "returns_rate_pct": 5})
    rules_text = st.text_area("JSON で設定", value=json.dumps(rules_json, ensure_ascii=False, indent=2), height=180)
    if st.button("保存（通知ルール）"):
        try:
            save_setting("notify_rules", json.loads(rules_text))
            st.success("保存しました")
        except Exception as e:
            st.error(f"JSON解析エラー: {e}")

    st.subheader("エクスポート")
    if st.button("勘定科目CSVマッピング表（雛形）"):
        df = pd.DataFrame({"account":["売上","仕入","送料","手数料","広告費","クーポン","ポイント","税金"],"column":["rev","cogs","ship_total","fee_total","ad_alloc_total","coupon_alloc","points_alloc","tax"]})
        st.download_button("download.csv", df_to_csv_bytes(df), file_name="account_mapping_template.csv")


def page_audit():
    st.header("監査ログ / 監視")
    eng = get_engine()
    try:
        logs = pd.read_sql_table("audit_logs", eng)
        logs = logs.sort_values("ts", ascending=False).head(2000)
        st.dataframe(logs)
        st.download_button("CSVエクスポート", df_to_csv_bytes(logs), file_name="audit_logs.csv")
    except Exception:
        st.info("ログがまだありません。")

# =============== 通知（SHOULD FR-104） ===============

def run_threshold_checks():
    rules = load_setting("notify_rules", {"gp_drop_pct": 20, "returns_rate_pct": 5})
    end = TODAY
    start = end - relativedelta(months=1)
    prev_start = start - relativedelta(months=1)
    prev_end = start
    df_now = get_profit_frame(start, end, [], [])
    df_prev = get_profit_frame(prev_start, prev_end, [], [])
    gp_now = float(df_now["gross_profit"].sum()) if not df_now.empty else 0
    gp_prev = float(df_prev["gross_profit"].sum()) if not df_prev.empty else 0
    drop = 100.0 * (gp_prev - gp_now) / gp_prev if gp_prev else 0
    rr = returns_rate(start, end) * 100

    texts = []
    if drop >= rules.get("gp_drop_pct", 20):
        texts.append(f"粗利急減: 前月比 -{drop:.1f}%")
    if rr >= rules.get("returns_rate_pct", 5):
        texts.append(f"返品率高止まり: {rr:.2f}%")
    if texts:
        send_slack("\n".join(texts))

# =============== メイン ===============

def main():
    st.set_page_config(page_title="EC収益管理", layout="wide")
    init_db()

    st.sidebar.title("EC収益管理（Streamlit）")
    login_panel()

    if not st.session_state.get("user"):
        st.info("左のパネルでログインしてください。初期は admin/admin など。")
        return

    company = load_setting("company_name", st.secrets.get("app", {}).get("company_name", "自社EC"))
    st.sidebar.markdown(f"**事業者：{company}**")

    # ページ選択（ロールに応じて制限）
    pages = [
        ("dashboard", "ダッシュボード", page_dashboard),
        ("import", "データ取込", page_import),
        ("inventory", "在庫管理", page_inventory),
        ("profit", "利益分析", page_profit),
        ("returns", "返品・不良", page_returns),
        ("rfm", "RFM/顧客", page_rfm),
        ("settings", "設定", page_settings),
        ("audit", "監査ログ", page_audit),
    ]
    avail = {k: v for k, v in ROLES.items()}  # 参照
    options = [label for key, label, _ in pages if role_allows(key)]
    label2func = {label: fn for key, label, fn in pages if role_allows(key)}
    choice = st.sidebar.radio("メニュー", options=options, index=0)
    log("nav", choice)

    # 自動しきい値チェック（簡易）
    if st.sidebar.button("しきい値チェック実行（Slack通知）"):
        run_threshold_checks()
        st.sidebar.success("チェック完了")

    # 描画
    label2func[choice]()

if __name__ == "__main__":
    main()
