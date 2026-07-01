# SQL Agent

Agentic natural language to SQL system built with LangGraph and Claude. Unlike a one-shot text-to-SQL tool, this agent executes its own queries, evaluates whether the result actually answers the question, and iterates with reflection until it's confident.

## How it works

1. User asks a question in plain English
2. Agent inspects the database schema
3. Agent writes a SQL query
4. Query is executed against the database
5. Agent evaluates: does this result answer the question?
6. If not, it reflects on what went wrong and writes a better query
7. Repeats up to 4 times, then returns the best answer

## Setup

```bash
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Create the sample database:

```bash
python src/database.py
```

Start the API:

```bash
uvicorn api.main:app --reload
```

Open `frontend/index.html` in your browser or visit `http://localhost:8000/app`.

## Sample questions

- Who are the top 3 customers by total spend?
- What is the best-selling product category?
- Which customers placed more than one order?
- What is the average order value by country?
- Which products have never been ordered?

## Stack

- **LangGraph** — agent loop with reflection
- **Claude** — SQL generation and result evaluation
- **SQLite** — sample e-commerce database
- **FastAPI** — REST API
