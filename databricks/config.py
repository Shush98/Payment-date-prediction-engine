"""Naming conventions for the lakehouse. Imported by every notebook.

Free Edition provisions a Unity Catalog metastore with a default catalog named
`workspace`; everything below lives in one schema inside it.
"""

CATALOG = "workspace"
SCHEMA = "payment_ops"

TABLES = {
    "bronze_events":     "bronze_invoice_events",
    "silver_invoices":   "silver_invoices",
    "silver_payments":   "silver_payments",
    "silver_outcomes":   "silver_invoice_outcomes",
    "quarantine":        "silver_quarantine",
    "dq_metrics":        "silver_data_quality_metrics",
    "gold_customer":     "gold_customer_payment_behavior",
    "feature_timeline":  "feat_customer_payment_history",
}

VOLUMES = {
    "raw":         "raw",          # dataset.csv is uploaded here once
    "landing":     "landing",      # generator drops event JSON here, Auto Loader reads it
    "checkpoints": "checkpoints",  # streaming state + schema inference
}


class Paths:
    """Resolves fully-qualified names so notebooks never hardcode catalog.schema."""

    def __init__(self, catalog=CATALOG, schema=SCHEMA):
        self.catalog = catalog
        self.schema = schema

    def table(self, key):
        return f"{self.catalog}.{self.schema}.{TABLES.get(key, key)}"

    def volume(self, key):
        return f"/Volumes/{self.catalog}/{self.schema}/{VOLUMES.get(key, key)}"

    def checkpoint(self, name):
        return f"{self.volume('checkpoints')}/{name}"

    def __repr__(self):
        return f"Paths({self.catalog}.{self.schema})"
