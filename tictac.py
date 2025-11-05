"""
Tic-Tac-Toe with Minimax (alpha–beta) — Streamlit App
-----------------------------------------------------
Run locally:
  1) pip install streamlit
  2) streamlit run ttt_minimax_app.py

Features
- Human vs AI (choose X/O)
- Difficulty: Easy / Medium / Hard (unbeatable)
- First move: You or AI
- Minimax with alpha–beta pruning and depth-based tie‑breakers
- Persistent session state, per‑session stats, and move history
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import streamlit as st

# -----------------------------
# Game logic helpers
# -----------------------------
Board = List[str]  # 9-length list, values in {"X","O",""}
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diagonals
]


def check_winner(board: Board) -> Optional[str]:
    """Return 'X' or 'O' if someone won, 'draw' if full and no winner, else None."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(board):
        return "draw"
    return None


def available_moves(board: Board) -> List[int]:
    return [i for i, v in enumerate(board) if not v]


@dataclass
class MiniResult:
    score: int
    move: Optional[int]


def minimax(board: Board, ai: str, human: str, maximizing: bool, depth: int, 
            alpha: int, beta: int, depth_limit: Optional[int]) -> MiniResult:
    """Minimax with alpha–beta.
    Scores: +10 for AI win, -10 for human win, 0 for draw. Depth is used to
    prefer quicker wins and slower losses (±(10 - depth)).
    If depth_limit is provided, search stops at that depth with a heuristic.
    """
    winner = check_winner(board)
    if winner == ai:
        return MiniResult(score=10 - depth, move=None)
    elif winner == human:
        return MiniResult(score=depth - 10, move=None)
    elif winner == "draw":
        return MiniResult(score=0, move=None)

    if depth_limit is not None and depth >= depth_limit:
        # Heuristic evaluation at cutoff: center > corners > edges + potential forks
        # Simple heuristic: count potential winning lines for each side
        def potential(player: str) -> int:
            count = 0
            for a, b, c in WIN_LINES:
                line = [board[a], board[b], board[c]]
                if all(v in ("", player) for v in line):
                    count += 1
            return count
        h = potential(ai) - potential(human)
        return MiniResult(score=h, move=None)

    best_move: Optional[int] = None

    if maximizing:
        best_score = -10_000
        for m in available_moves(board):
            board[m] = ai
            res = minimax(board, ai, human, False, depth + 1, alpha, beta, depth_limit)
            board[m] = ""
            if res.score > best_score:
                best_score, best_move = res.score, m
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return MiniResult(best_score, best_move)
    else:
        best_score = 10_000
        for m in available_moves(board):
            board[m] = human
            res = minimax(board, ai, human, True, depth + 1, alpha, beta, depth_limit)
            board[m] = ""
            if res.score < best_score:
                best_score, best_move = res.score, m
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return MiniResult(best_score, best_move)


# -----------------------------
# AI move selection per difficulty
# -----------------------------

def ai_best_move(board: Board, ai: str, human: str, difficulty: str) -> int:
    moves = available_moves(board)
    if not moves:
        return -1

    # Opening book for unbeatable play: take center if free, else a corner
    if difficulty == "Hard":
        if 4 in moves:
            return 4
        for c in [0, 2, 6, 8]:
            if c in moves:
                return c

    if difficulty == "Easy":
        # 60% random, 40% shallow minimax
        if random.random() < 0.6:
            return random.choice(moves)
        depth_limit = 1
    elif difficulty == "Medium":
        depth_limit = 3
    else:  # Hard
        depth_limit = None

    res = minimax(board, ai, human, True, depth=0, alpha=-10_000, beta=10_000, depth_limit=depth_limit)
    return res.move if res.move is not None else random.choice(moves)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Tic‑Tac‑Toe — Minimax AI", page_icon="🎮", layout="centered")
st.title("Tic‑Tac‑Toe — Minimax AI 🎮")
st.caption("Human vs. AI with alpha–beta pruning. Built in Streamlit.")

# Init session state
if "board" not in st.session_state:
    st.session_state.board = [""] * 9
if "player_symbol" not in st.session_state:
    st.session_state.player_symbol = "X"
if "ai_symbol" not in st.session_state:
    st.session_state.ai_symbol = "O"
if "turn" not in st.session_state:
    st.session_state.turn = "player"  # or "ai"
if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Hard"
if "history" not in st.session_state:
    st.session_state.history = []  # list of (move_index, symbol)
if "stats" not in st.session_state:
    st.session_state.stats = {"wins": 0, "losses": 0, "draws": 0}


def reset_board(new_game: bool = False):
    st.session_state.board = [""] * 9
    st.session_state.history = []
    if new_game:
        # preserve stats, symbols, difficulty
        pass


# Sidebar controls
with st.sidebar:
    st.header("Settings")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=["Easy","Medium","Hard"].index(st.session_state.difficulty))
    if difficulty != st.session_state.difficulty:
        st.session_state.difficulty = difficulty

    colA, colB = st.columns(2)
    with colA:
        chosen = st.radio("You play as", ["X", "O"], index=0 if st.session_state.player_symbol == "X" else 1, horizontal=True)
    with colB:
        first = st.radio("First move", ["You", "AI"], index=0 if st.session_state.turn == "player" else 1, horizontal=True)

    if chosen != st.session_state.player_symbol:
        st.session_state.player_symbol = chosen
        st.session_state.ai_symbol = "O" if chosen == "X" else "X"
        reset_board()

    if (first == "You" and st.session_state.turn != "player") or (first == "AI" and st.session_state.turn != "ai"):
        st.session_state.turn = "player" if first == "You" else "ai"
        reset_board()

    if st.button("🔄 New Game"):
        reset_board(new_game=True)

    st.markdown("---")
    st.subheader("Session stats")
    s = st.session_state.stats
    st.metric("Wins", s["wins"]) 
    st.metric("Losses", s["losses"]) 
    st.metric("Draws", s["draws"]) 


# Auto AI move if it's AI's turn
outcome = check_winner(st.session_state.board)
if st.session_state.turn == "ai" and outcome is None:
    idx = ai_best_move(st.session_state.board, st.session_state.ai_symbol, st.session_state.player_symbol, st.session_state.difficulty)
    if idx != -1 and st.session_state.board[idx] == "":
        st.session_state.board[idx] = st.session_state.ai_symbol
        st.session_state.history.append((idx, st.session_state.ai_symbol))
        st.session_state.turn = "player"
    outcome = check_winner(st.session_state.board)


# Board grid UI
st.subheader("Board")

board = st.session_state.board
clicked_index = None
for r in range(3):
    cols = st.columns(3)
    for c in range(3):
        i = r * 3 + c
        label = board[i] if board[i] else "\u2800"  # braille blank keeps button size
        disabled = bool(board[i]) or check_winner(board) is not None or st.session_state.turn != "player"
        with cols[c]:
            if st.button(label, key=f"cell_{i}", use_container_width=True, disabled=disabled):
                clicked_index = i

# Handle human click
if clicked_index is not None and board[clicked_index] == "" and st.session_state.turn == "player" and check_winner(board) is None:
    board[clicked_index] = st.session_state.player_symbol
    st.session_state.history.append((clicked_index, st.session_state.player_symbol))
    st.session_state.turn = "ai"

# After potential human move, let AI play immediately (rerun)
outcome = check_winner(board)
if st.session_state.turn == "ai" and outcome is None:
    st.rerun()

# Outcome banner
outcome = check_winner(board)
if outcome:
    if outcome == "draw":
        st.success("It's a draw!")
        st.session_state.stats["draws"] += 1
    elif outcome == st.session_state.player_symbol:
        st.success("You win! 🏆")
        st.session_state.stats["wins"] += 1
    else:
        st.error("AI wins! 🤖")
        st.session_state.stats["losses"] += 1

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Play again"):
            reset_board(new_game=True)
            st.rerun()
    with col2:
        if st.button("Reset & switch first move"):
            st.session_state.turn = "player" if st.session_state.turn == "ai" else "ai"
            reset_board(new_game=True)
            st.rerun()

# Move history
st.markdown("---")
with st.expander("Move history"):
    pretty = [f"{i+1}. {'Row ' + str((idx//3)+1) + ', Col ' + str((idx%3)+1)} — {sym}" for i, (idx, sym) in enumerate(st.session_state.history)]
    if pretty:
        st.write("\n".join(pretty))
    else:
        st.caption("No moves yet.")

# Footer
st.markdown(
    """
    <div style="text-align:center; font-size:0.9rem; opacity:0.7; margin-top: 1rem;">
        Built for an AI course project • Minimax + alpha–beta • Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
