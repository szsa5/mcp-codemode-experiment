"""Task definitions for the synthetic ground.

A "question" is one concrete task instance: a task type, a start node, a depth,
the natural-language prompt handed to the model, and the oracle answer. The
sweep runner crosses these questions with the arms (standard MCP, Code Mode),
the tool count K (decoys), and the surface (coarse, fine).

Two task types:
- final_value (RQ1 depth/K sweep): follow next_id D times, report the final
  node's value. One read per hop suffices on either surface.
- value_sum (RQ2 granularity): follow next_id D times and sum the value of every
  visited node. Needs each node's value and its next pointer, so the fine
  surface pays two calls per hop where the coarse surface pays one.
"""

from dataclasses import dataclass

import graph_data


@dataclass(frozen=True)
class Question:
    task_id: str
    task_type: str  # "final_value" | "value_sum"
    start_id: str
    depth: int
    prompt: str
    expected: int


_PREAMBLE = (
    "You are working with a directed graph of nodes. Each node has a string id, "
    "an integer `value`, an `attrs` object, and a `next_id` pointer to the next "
    "node in the chain. You can only learn a node's contents by calling the "
    "provided tools; never guess or invent values."
)


def _final_value_prompt(start_id: str, depth: int) -> str:
    return (
        f"{_PREAMBLE}\n\n"
        f"Starting at node '{start_id}', follow the `next_id` pointer exactly "
        f"{depth} time(s) to reach the destination node. "
        f"Report ONLY the integer `value` stored at that destination node."
    )


def _value_sum_prompt(start_id: str, depth: int) -> str:
    total_nodes = depth + 1
    return (
        f"{_PREAMBLE}\n\n"
        f"Starting at node '{start_id}', visit that node and then follow the "
        f"`next_id` pointer {depth} time(s), visiting {total_nodes} nodes in "
        f"total. Read each visited node's `value`. "
        f"Report ONLY the integer sum of the `value` fields of all "
        f"{total_nodes} visited nodes."
    )


def rq1_questions(graph: dict, depths: list[int], repeats: int) -> list[Question]:
    """One final_value question per (depth, repeat). Repeat r starts at node nr."""
    questions: list[Question] = []
    for depth in depths:
        for r in range(repeats):
            start_id = f"n{r}"
            questions.append(
                Question(
                    task_id=f"final_value_d{depth}_r{r}",
                    task_type="final_value",
                    start_id=start_id,
                    depth=depth,
                    prompt=_final_value_prompt(start_id, depth),
                    expected=graph_data.oracle_final_value(graph, start_id, depth),
                )
            )
    return questions


def rq2_questions(graph: dict, depth: int, repeats: int) -> list[Question]:
    """One value_sum question per repeat at a fixed depth. Repeat r starts at nr."""
    questions: list[Question] = []
    for r in range(repeats):
        start_id = f"n{r}"
        questions.append(
            Question(
                task_id=f"value_sum_d{depth}_r{r}",
                task_type="value_sum",
                start_id=start_id,
                depth=depth,
                prompt=_value_sum_prompt(start_id, depth),
                expected=graph_data.oracle_value_sum(graph, start_id, depth),
            )
        )
    return questions
