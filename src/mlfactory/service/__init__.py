"""FastAPI service exposing a trained AlphaZero checkpoint as an HTTP bot.

Designed to be called by the Boop TS server (or any other front-end) as:

    POST /move
        {"state": <boop GameState JSON>, "color": "orange"|"gray"}
    ->  {"kind": "place", "row": 3, "col": 4, "pieceType": "kitten"}
    or
        {"kind": "graduation", "optionIndex": 0}
"""
