"""Training: self-play loop, replay buffer, and the AlphaZero trainer.

Modules:
- replay_buffer.py : bounded FIFO of (state-planes, policy-target, value-target)
- selfplay.py      : generate training data by having the net play itself
- sample_game.py   : serialize a played game to disk (human + machine readable)
- trainer.py       : main subprocess entry point — orchestrates the loop
"""
