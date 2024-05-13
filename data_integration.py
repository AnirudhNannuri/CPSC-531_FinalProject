# data_integration.py
def integrate_data(client_df, invoice_df):
    """Join client and invoice data on client_id."""
    return client_df.join(invoice_df, "client_id")
