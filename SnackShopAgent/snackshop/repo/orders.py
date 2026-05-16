
from data.orders import ORDER_DATABASE


def find_order(key: str) -> tuple[str | None, dict | None]:
    """Search ORDER_DATABASE by order ID, tracking ID, or email.

    Returns (order_id, order_dict) or (None, None) if not found.
    """
    key = key.strip()

    # 1. Direct order ID match (e.g. ORD-201)
    upper_key = key.upper()
    if upper_key in ORDER_DATABASE:
        return upper_key, ORDER_DATABASE[upper_key]

    # 2. Search by tracking ID or email
    for oid, order in ORDER_DATABASE.items():
        if order["tracking_id"].upper() == upper_key:
            return oid, order
        if order["customer_email"].lower() == key.lower():
            return oid, order

    return None, None

def format_order(order_id: str, order: dict) -> str:
    """Format order details into a readable string."""
    return (
        f"Order {order_id} Details:\n"
        f"  Item       : {order['item_name']}\n"
        f"  Customer   : {order['customer_name']}\n"
        f"  Email      : {order['customer_email']}\n"
        f"  Status     : {order['status']}\n"
        f"  Price      : ₹{order['price']}\n"
        f"  Ordered on : {order['order_date']}\n"
        f"  Est. delivery: {order['estimated_delivery']}\n"
        f"  Tracking ID: {order['tracking_id']}"
    )