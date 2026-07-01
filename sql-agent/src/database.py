import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "ecommerce.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def get_schema() -> str:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)

        cursor.execute(f"SELECT * FROM {table} LIMIT 2")
        samples = cursor.fetchall()
        col_names = [c[1] for c in cols]

        schema_parts.append(
            f"Table: {table}({col_defs})\n"
            f"Sample rows: {[dict(zip(col_names, row)) for row in samples]}"
        )

    conn.close()
    return "\n\n".join(schema_parts)


def execute_sql(query: str) -> dict:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchmany(50)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        conn.close()
        return {"columns": columns, "rows": rows, "error": None}
    except Exception as e:
        return {"columns": [], "rows": [], "error": str(e)}


def setup_database():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = get_connection()
    c = conn.cursor()

    c.executescript("""
    DROP TABLE IF EXISTS order_items;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        email TEXT,
        city TEXT,
        country TEXT,
        signup_date TEXT
    );

    CREATE TABLE products (
        id INTEGER PRIMARY KEY,
        name TEXT,
        category TEXT,
        price REAL,
        stock INTEGER
    );

    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        order_date TEXT,
        status TEXT,
        total_amount REAL
    );

    CREATE TABLE order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER REFERENCES orders(id),
        product_id INTEGER REFERENCES products(id),
        quantity INTEGER,
        unit_price REAL
    );
    """)

    customers = [
        (1, "Alice Chen",     "alice@email.com",   "New York",    "USA",     "2022-01-15"),
        (2, "Bob Smith",      "bob@email.com",      "London",      "UK",      "2022-03-22"),
        (3, "Carlos Rivera",  "carlos@email.com",   "Toronto",     "Canada",  "2022-05-10"),
        (4, "Diana Park",     "diana@email.com",    "Sydney",      "Australia","2022-07-04"),
        (5, "Ethan James",    "ethan@email.com",    "New York",    "USA",     "2022-09-18"),
        (6, "Fatima Al-Zahra","fatima@email.com",   "Dubai",       "UAE",     "2023-01-02"),
        (7, "George Lee",     "george@email.com",   "Singapore",   "Singapore","2023-02-14"),
        (8, "Hannah White",   "hannah@email.com",   "Chicago",     "USA",     "2023-04-30"),
        (9, "Ivan Petrov",    "ivan@email.com",     "Berlin",      "Germany", "2023-06-11"),
        (10,"Julia Santos",   "julia@email.com",    "São Paulo",   "Brazil",  "2023-08-25"),
    ]
    c.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", customers)

    products = [
        (1,  "Laptop Pro 15",      "Electronics",  1299.99, 45),
        (2,  "Wireless Headphones","Electronics",   199.99, 120),
        (3,  "Running Shoes",      "Sports",        89.99,  200),
        (4,  "Coffee Maker",       "Kitchen",       79.99,  85),
        (5,  "Python Cookbook",    "Books",         39.99,  300),
        (6,  "Standing Desk",      "Furniture",    449.99,  30),
        (7,  "Yoga Mat",           "Sports",        35.99,  150),
        (8,  "Mechanical Keyboard","Electronics",  149.99,  75),
        (9,  "Air Purifier",       "Home",         249.99,  60),
        (10, "Smartwatch",         "Electronics",  299.99,  90),
        (11, "Blender",            "Kitchen",       59.99, 110),
        (12, "Desk Lamp",          "Furniture",     45.99, 200),
    ]
    c.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    orders = [
        (1,  1, "2024-01-05", "delivered", 1499.98),
        (2,  2, "2024-01-12", "delivered",  199.99),
        (3,  3, "2024-01-20", "delivered",  169.98),
        (4,  1, "2024-02-03", "delivered",  449.99),
        (5,  4, "2024-02-14", "delivered",  299.99),
        (6,  5, "2024-02-28", "shipped",    239.98),
        (7,  6, "2024-03-05", "delivered", 1299.99),
        (8,  7, "2024-03-11", "delivered",  485.98),
        (9,  2, "2024-03-19", "cancelled",  89.99),
        (10, 8, "2024-04-02", "delivered",  329.98),
        (11, 3, "2024-04-15", "shipped",    149.99),
        (12, 9, "2024-04-22", "delivered",  709.98),
        (13, 1, "2024-05-01", "delivered",  339.98),
        (14,10, "2024-05-10", "pending",    249.99),
        (15, 5, "2024-05-18", "delivered",  199.99),
    ]
    c.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)

    order_items = [
        (1,  1,  1, 1, 1299.99),(2,  1,  2, 1,  199.99),
        (3,  2,  2, 1,  199.99),(4,  3,  3, 1,   89.99),
        (5,  3,  7, 2,   35.99),(6,  4,  6, 1,  449.99),
        (7,  5, 10, 1,  299.99),(8,  6,  8, 1,  149.99),
        (9,  6, 12, 2,   45.99),(10, 7,  1, 1, 1299.99),
        (11, 8,  6, 1,  449.99),(12, 8,  4, 1,   79.99),
        (13, 9,  3, 1,   89.99),(14,10,  9, 1,  249.99),
        (15,10, 11, 1,   59.99),(16,11,  8, 1,  149.99),
        (17,12,  1, 1, 1299.99),(18,12, 10, 1,  299.99),
        (19,12, 12, 3,   45.99),(20,13,  2, 1,  199.99),
        (21,13, 10, 1,  299.99),(22,14,  9, 1,  249.99),
        (23,15,  2, 1,  199.99),
    ]
    c.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", order_items)

    conn.commit()
    conn.close()
    print(f"Database created at {DB_PATH}")


if __name__ == "__main__":
    setup_database()
