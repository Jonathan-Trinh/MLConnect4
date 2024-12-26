from dataclasses import dataclass

@dataclass
class Piece:
    turn: int
    coords: tuple