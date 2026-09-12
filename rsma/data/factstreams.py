"""
North star 1 training signal: dialogues where facts are told early and asked about later, past
the attention window, so the loss on the answers teaches the fast weights to carry meaning.

Each stream is n_seq windows of seq_len tokens. Window 0 tells K facts inside ordinary
conversation filler; later windows ask about them (and about other things, as filler). Facts use
templates and random values so the model cannot guess. Formatted in the base's chat format.
"""
import random

import numpy as np
import torch

TEMPLATES = [
    ("my dog's name", "What is my dog's name?", ["Biscuit", "Pepper", "Juno", "Maple", "Otis", "Rex", "Luna", "Ziggy"]),
    ("the city I was born in", "Which city was I born in?", ["Philadelphia", "Atlanta", "Detroit", "Houston", "Oakland", "Denver", "Memphis", "Boston"]),
    ("my favorite composer", "Who is my favorite composer?", ["Coltrane", "Bach", "Nina Simone", "Ellington", "Debussy", "Mingus", "Chopin", "Monk"]),
    ("the month of my birthday", "What month is my birthday?", ["March", "August", "November", "June", "January", "October", "April", "July"]),
    ("the language I am learning", "Which language am I learning?", ["Portuguese", "Japanese", "Swahili", "Arabic", "Korean", "Italian", "Yoruba", "Greek"]),
    ("my sister's profession", "What does my sister do for work?", ["a nurse", "an architect", "a pilot", "a chef", "a lawyer", "a teacher", "a welder", "a dentist"]),
    ("the car I drive", "What car do I drive?", ["a Volvo", "a Honda", "a Tesla", "a Subaru", "a Jeep", "a Ford", "a Toyota", "a Mazda"]),
    ("my favorite book", "What is my favorite book?", ["Invisible Man", "Dune", "Beloved", "Meditations", "The Odyssey", "Walden", "Kindred", "Ulysses"]),
    ("the sport I played in college", "Which sport did I play in college?", ["rugby", "swimming", "basketball", "fencing", "track", "soccer", "rowing", "tennis"]),
    ("the street I live on", "What street do I live on?", ["Cedar Street", "Lincoln Avenue", "Baker Road", "Vine Street", "Harbor Lane", "Elm Court", "Pine Way", "Oak Drive"]),
    ("my daughter's name", "What is my daughter's name?", ["Amara", "Zora", "Ines", "Nia", "Thandi", "Ada", "Imani", "Sade"]),
    ("the instrument I play", "Which instrument do I play?", ["the cello", "the trumpet", "the piano", "the kora", "the drums", "the guitar", "the saxophone", "the violin"]),
    ("my company's name", "What is my company called?", ["Northwind", "Blue Heron", "Kestrel Labs", "Sankofa Works", "Ironwood", "Lantern", "Redline", "Harbor Point"]),
    ("the number of my apartment", "What is my apartment number?", ["4B", "12C", "7A", "22F", "3D", "9E", "15G", "1H"]),
]

FILLER_Q = [
    "What did Frederick Douglass say about struggle?", "How does compound interest work?", "What is a delta rule?",
    "Why did Rome fall?", "How should I start meditating?", "What makes a good teacher?", "What is Bitcoin's supply limit?",
    "Who was Ada Lovelace?", "What is the Fourteenth Amendment about?", "How do I write clearly?", "What did Darwin observe on the Beagle?",
    "What is Stoicism?", "How do you value a business?", "What was the Harlem Renaissance?", "What is natural selection?",
]
FILLER_A = [
    "That is a question with a long history, and the short version is this:", "Let me answer that directly.",
    "Here is what I know.", "It depends on what you mean, but broadly:", "There are two parts to it.",
]


class FactStreams:
    """Yields (x, y) windows in chat format with the base tokenizer. Loss targets are the whole
    stream (the answer tokens are where the memory signal lives)."""

    def __init__(self, tokenizer, system_prompt, seq_len=512, seed=0, device="cpu", facts_per_stream=3):
        self.tok = tokenizer
        self.system_prompt = system_prompt
        self.seq_len = seq_len
        self.rng = random.Random(seed)
        self.device = device
        self.k = facts_per_stream
        self.vocab = len(tokenizer)

    def _turn(self, role, content):
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def _stream_text(self, n_seq):
        r = self.rng
        facts = r.sample(TEMPLATES, self.k)
        values = {t[0]: r.choice(t[2]) for t in facts}
        parts = [self._turn("system", self.system_prompt)]
        # window 0: tell the facts, interleaved with filler
        tell = [self._turn("user", f"Something about me: {t[0]} is {values[t[0]]}.") + self._turn("assistant", r.choice(["Noted.", "I will remember that.", "Understood.", "Good to know."])) for t in facts]
        fill = [self._turn("user", q) + self._turn("assistant", r.choice(FILLER_A) + " " + q.lower().replace("?", ".") + " " * 3) for q in r.sample(FILLER_Q, 4)]
        seq = tell + fill
        r.shuffle(seq)
        parts += seq
        # later: ask about them among filler, repeatedly
        for _ in range(n_seq * 2):
            if r.random() < 0.5:
                t = r.choice(facts)
                parts.append(self._turn("user", t[1]) + self._turn("assistant", f"{values[t[0]]}."))
            else:
                q = r.choice(FILLER_Q)
                parts.append(self._turn("user", q) + self._turn("assistant", r.choice(FILLER_A)))
        return "".join(parts)

    def stream(self, batch, n_seq, split="train"):
        T = self.seq_len
        need = n_seq * T + 1
        rows = []
        for _ in range(batch):
            ids = []
            while len(ids) < need:
                ids += self.tok(self._stream_text(n_seq), add_special_tokens=False).input_ids
            rows.append(ids[:need])
        arr = torch.tensor(rows, device=self.device)
        for s in range(n_seq):
            yield arr[:, s * T:(s + 1) * T], arr[:, s * T + 1:(s + 1) * T + 1]

    def batch(self, batch, split="train"):
        return next(iter(self.stream(batch, 1, split)))
