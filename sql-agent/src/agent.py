import os
import re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from groq import Groq
from database import get_schema, execute_sql

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 4


class AgentState(TypedDict):
    question: str
    schema: str
    history: list       # list of {"query": str, "result": dict, "reflection": str}
    current_query: str
    current_result: dict
    final_answer: str
    iterations: int
    done: bool


def _call_llm(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _extract_sql(text: str) -> str:
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: take first SELECT statement
    match = re.search(r"(SELECT .+?)(?:;|$)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def write_query_node(state: AgentState) -> AgentState:
    history_text = ""
    for h in state["history"]:
        history_text += (
            f"\nAttempt:\nSQL: {h['query']}\n"
            f"Result: {h['result']}\n"
            f"Reflection: {h['reflection']}\n"
        )

    system = (
        "You are an expert SQL assistant. Write a SQLite query to answer the user's question. "
        "Return ONLY the SQL inside a ```sql ``` block, nothing else."
    )
    user = (
        f"Database schema:\n{state['schema']}\n\n"
        f"Question: {state['question']}\n"
        + (f"\nPrevious failed attempts:{history_text}\nWrite an improved query." if history_text else "")
    )

    text = _call_llm(system, user)
    return {**state, "current_query": _extract_sql(text)}


def execute_query_node(state: AgentState) -> AgentState:
    result = execute_sql(state["current_query"])
    return {**state, "current_result": result}


def evaluate_node(state: AgentState) -> AgentState:
    result = state["current_result"]

    if result["error"]:
        reflection = f"SQL error: {result['error']}. Fix the query syntax or logic."
        history = state["history"] + [{
            "query": state["current_query"],
            "result": result,
            "reflection": reflection,
        }]
        return {**state, "history": history, "iterations": state["iterations"] + 1, "done": False}

    result_preview = {"columns": result["columns"], "rows": result["rows"][:5]}

    system = (
        "You are evaluating whether a SQL query result fully answers a user's question. "
        "Reply with either:\n"
        "ANSWER: <natural language answer based on the results>\n"
        "or\n"
        "RETRY: <one sentence explaining what is wrong or missing>"
    )
    user = (
        f"Question: {state['question']}\n"
        f"SQL: {state['current_query']}\n"
        f"Result: {result_preview}"
    )

    response = _call_llm(system, user)

    if response.startswith("ANSWER:") or state["iterations"] + 1 >= MAX_ITERATIONS:
        answer = response.replace("ANSWER:", "").strip()
        if not answer:
            answer = f"Query returned {len(result['rows'])} rows across columns: {result['columns']}"
        return {**state, "final_answer": answer, "iterations": state["iterations"] + 1, "done": True}

    reflection = response.replace("RETRY:", "").strip()
    history = state["history"] + [{
        "query": state["current_query"],
        "result": result_preview,
        "reflection": reflection,
    }]
    return {**state, "history": history, "iterations": state["iterations"] + 1, "done": False}


def route(state: AgentState) -> str:
    return END if state["done"] else "write_query"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("write_query", write_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("evaluate", evaluate_node)

    graph.set_entry_point("write_query")
    graph.add_edge("write_query", "execute_query")
    graph.add_edge("execute_query", "evaluate")
    graph.add_conditional_edges("evaluate", route, {"write_query": "write_query", END: END})

    return graph.compile()


agent = build_graph()


def run(question: str) -> dict:
    schema = get_schema()
    final_state = agent.invoke({
        "question": question,
        "schema": schema,
        "history": [],
        "current_query": "",
        "current_result": {},
        "final_answer": "",
        "iterations": 0,
        "done": False,
    })

    return {
        "question": question,
        "answer": final_state["final_answer"],
        "final_query": final_state["current_query"],
        "iterations": final_state["iterations"],
        "history": final_state["history"],
    }
