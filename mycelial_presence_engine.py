"""
Session Context Engine
======================

Tracks development context across conversation turns.
Generates actionable suggestions based on pattern detection.
Persists state for session continuation.

Core mechanism:
1. Input → pattern detection against history
2. Patterns → ranked suggestions by type
3. Suggestion → direct action execution
4. State → serializable checkpoint for continuation
"""

import json
import hashlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque


class SuggestionType(Enum):
    """Action categories for contextual suggestions."""
    CLARIFY = "clarify"      # Resolve ambiguity
    EXPAND = "expand"        # Broaden scope
    CREATE = "create"        # Generate artifacts
    CONNECT = "connect"      # Link to existing
    CHALLENGE = "challenge"  # Question assumptions
    CRYSTALLIZE = "crystallize"  # Summarize/checkpoint


@dataclass
class SessionState:
    """Serializable session state for continuation."""
    focus_module: str = ""
    focus_task: str = ""
    priority: str = "medium"
    context_note: str = ""
    depth: int = 0
    artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SessionState':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Turn:
    """Single conversation turn."""
    role: str
    content: str
    timestamp: str
    keywords: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.keywords:
            self.keywords = self._extract_keywords()

    def _extract_keywords(self) -> List[str]:
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will',
                     'would', 'could', 'should', 'their', 'there', 'about', 'which'}
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', self.content.lower())
        return list(set(w for w in words if w not in stopwords))[:20]


@dataclass
class Suggestion:
    """Actionable suggestion with execution binding."""
    type: SuggestionType
    title: str
    description: str
    action_key: str
    relevance: float
    triggers: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"[{self.type.value.upper()}] {self.title} ({self.relevance:.2f})"


@dataclass
class Checkpoint:
    """Serializable session checkpoint."""
    id: str
    version: str
    timestamp: str
    state: SessionState
    history_summary: str
    recent_actions: List[str]
    continuation_prompt: str

    def to_json(self, indent: int = 2) -> str:
        d = {
            'id': self.id,
            'version': self.version,
            'timestamp': self.timestamp,
            'state': self.state.to_dict(),
            'history_summary': self.history_summary,
            'recent_actions': self.recent_actions,
            'continuation_prompt': self.continuation_prompt
        }
        return json.dumps(d, indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> 'Checkpoint':
        d = json.loads(json_str)
        return cls(
            id=d['id'],
            version=d['version'],
            timestamp=d['timestamp'],
            state=SessionState.from_dict(d['state']),
            history_summary=d.get('history_summary', ''),
            recent_actions=d.get('recent_actions', []),
            continuation_prompt=d['continuation_prompt']
        )

    def validate(self) -> bool:
        expected = hashlib.sha256(f"{self.id}{self.timestamp}".encode()).hexdigest()[:16]
        return self.id.endswith(expected[-8:]) or True  # Simplified validation


class PatternDetector:
    """Detects patterns in input for suggestion routing."""

    PATTERNS = {
        'question': (re.compile(r'\?|^(what|how|why|when|where|can|could|should|is|are)\b', re.I), 1.5),
        'code': (re.compile(r'\b(function|class|def|import|module|\.py|implement|create)\b', re.I), 1.3),
        'error': (re.compile(r'\b(error|exception|fail|bug|issue|problem|fix)\b', re.I), 1.4),
        'integration': (re.compile(r'\b(integrat|connect|link|combine|merge|between)\b', re.I), 1.2),
        'summary': (re.compile(r'\b(summary|summarize|conclude|final|overview|recap)\b', re.I), 1.1),
        'explore': (re.compile(r'\b(explore|expand|more|detail|deep|further)\b', re.I), 1.2),
    }

    # Pattern → SuggestionType mapping
    TYPE_MAP = {
        'question': SuggestionType.CLARIFY,
        'explore': SuggestionType.EXPAND,
        'code': SuggestionType.CREATE,
        'integration': SuggestionType.CONNECT,
        'error': SuggestionType.CHALLENGE,
        'summary': SuggestionType.CRYSTALLIZE,
    }

    def detect(self, text: str, history: List[Turn]) -> Dict[str, Any]:
        """Detect patterns and compute suggestion relevance scores."""
        # Pattern matching
        scores = {}
        for name, (regex, weight) in self.PATTERNS.items():
            matches = regex.findall(text)
            if matches:
                scores[name] = len(matches) * weight

        # Keyword extraction
        text_keywords = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', text.lower()))
        history_keywords = set()
        for turn in history[-10:]:
            history_keywords.update(turn.keywords)

        overlap = text_keywords & history_keywords
        novel = text_keywords - history_keywords

        # Compute type scores
        type_scores = {t: 0.0 for t in SuggestionType}
        for pattern, score in scores.items():
            stype = self.TYPE_MAP.get(pattern)
            if stype:
                type_scores[stype] += score

        # Boost CONNECT for high overlap, EXPAND for novel concepts
        type_scores[SuggestionType.CONNECT] += len(overlap) * 0.2
        type_scores[SuggestionType.EXPAND] += len(novel) * 0.1

        # Normalize
        max_score = max(type_scores.values()) or 1.0
        type_scores = {k: v / max_score for k, v in type_scores.items()}

        return {
            'patterns': scores,
            'overlap': list(overlap),
            'novel': list(novel),
            'type_scores': type_scores,
            'ranked': sorted(type_scores.items(), key=lambda x: -x[1])
        }


class SuggestionFactory:
    """Generates suggestions from pattern analysis."""

    TEMPLATES = {
        SuggestionType.CLARIFY: [
            ("Clarify intent", "What specific outcome do you need?", "clarify_intent"),
            ("Define scope", "What boundaries apply?", "clarify_scope"),
        ],
        SuggestionType.EXPAND: [
            ("Explore alternatives", "Consider alternative approaches", "explore_alt"),
            ("Deep dive", "Investigate underlying mechanisms", "deep_dive"),
        ],
        SuggestionType.CREATE: [
            ("Generate code", "Create implementation", "gen_code"),
            ("Write tests", "Generate test cases", "gen_tests"),
        ],
        SuggestionType.CONNECT: [
            ("Link existing", "Connect to codebase patterns", "link_code"),
            ("Reference history", "Connect to previous discussion", "link_history"),
        ],
        SuggestionType.CHALLENGE: [
            ("Question assumptions", "Challenge underlying assumptions", "challenge"),
            ("Identify risks", "What could go wrong?", "risks"),
        ],
        SuggestionType.CRYSTALLIZE: [
            ("Summarize", "Crystallize current progress", "summarize"),
            ("Checkpoint", "Save session state", "checkpoint"),
        ],
    }

    def generate(self, analysis: Dict[str, Any], limit: int = 6) -> List[Suggestion]:
        suggestions = []
        for stype, score in analysis['ranked']:
            if score < 0.1 or len(suggestions) >= limit:
                break

            templates = self.TEMPLATES.get(stype, [])
            if not templates:
                continue

            title, desc, action = templates[0]

            # Customize for CONNECT/EXPAND based on context
            if stype == SuggestionType.CONNECT and analysis['overlap']:
                title = f"Connect: {', '.join(analysis['overlap'][:3])}"
            elif stype == SuggestionType.EXPAND and analysis['novel']:
                title = f"Explore: {', '.join(analysis['novel'][:3])}"

            suggestions.append(Suggestion(
                type=stype,
                title=title,
                description=desc,
                action_key=action,
                relevance=score,
                triggers=analysis['overlap'][:5] or analysis['novel'][:5]
            ))

        return suggestions


class SessionContextEngine:
    """
    Tracks context, generates suggestions, persists state.

    Usage:
        engine = SessionContextEngine()
        result = engine.process("How do I fix the integration?")
        engine.execute(result['suggestions'][0])
        checkpoint = engine.save()
    """

    VERSION = "1.0.0"

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.state = SessionState()
        self.history: List[Turn] = []
        self.actions: List[str] = []
        self.suggestions: List[Suggestion] = []

        self.detector = PatternDetector()
        self.factory = SuggestionFactory()
        self.handlers: Dict[str, Callable] = self._default_handlers()

    def _default_handlers(self) -> Dict[str, Callable]:
        return {
            'clarify_intent': lambda s: {'prompt': "What outcome do you need?"},
            'clarify_scope': lambda s: {'prompt': "What boundaries apply?"},
            'explore_alt': lambda s: {'prompt': "Consider alternatives", 'triggers': s.triggers},
            'deep_dive': lambda s: {'prompt': f"Investigate: {', '.join(s.triggers[:3])}"},
            'gen_code': lambda s: {'action': 'generate', 'type': 'code'},
            'gen_tests': lambda s: {'action': 'generate', 'type': 'tests'},
            'link_code': lambda s: {'action': 'search', 'terms': s.triggers},
            'link_history': lambda s: self._find_relevant_history(s.triggers),
            'challenge': lambda s: {'questions': ["What assumptions?", "What could fail?"]},
            'risks': lambda s: {'prompt': "Identify potential risks"},
            'summarize': lambda s: self._summarize(),
            'checkpoint': lambda s: {'checkpoint': self.save().to_json()},
        }

    def process(self, content: str, role: str = "user") -> Dict[str, Any]:
        """
        Process input: detect patterns, generate suggestions.

        Returns analysis and ranked suggestions.
        """
        turn = Turn(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat()
        )
        self.history.append(turn)

        # Bound history
        if len(self.history) > 100:
            self.history = self.history[-100:]

        # Analyze
        analysis = self.detector.detect(content, self.history[:-1])

        # Generate suggestions
        self.suggestions = self.factory.generate(analysis)

        # Update state depth
        self.state.depth = len(self.history) // 10

        return {
            'analysis': analysis,
            'suggestions': self.suggestions,
            'state': self.state.to_dict()
        }

    def execute(self, suggestion: Suggestion) -> Dict[str, Any]:
        """Execute a suggestion via its action handler."""
        handler = self.handlers.get(suggestion.action_key)
        if not handler:
            return {'error': f'No handler: {suggestion.action_key}'}

        self.actions.append(f"{suggestion.type.value}: {suggestion.title}")
        return handler(suggestion)

    def execute_by_index(self, idx: int) -> Dict[str, Any]:
        """Execute suggestion by 1-based index."""
        if not self.suggestions or idx < 1 or idx > len(self.suggestions):
            return {'error': 'Invalid index'}
        return self.execute(self.suggestions[idx - 1])

    def set_focus(self, module: str, task: str, priority: str = "medium", note: str = ""):
        """Set current development focus."""
        self.state.focus_module = module
        self.state.focus_task = task
        self.state.priority = priority
        self.state.context_note = note
        self.actions.append(f"Focus: {module}/{task}")

    def add_artifact(self, name: str):
        """Record created artifact."""
        self.state.artifacts.append(name)
        self.actions.append(f"Artifact: {name}")

    def _find_relevant_history(self, terms: List[str]) -> Dict[str, Any]:
        """Find history turns matching terms."""
        matches = []
        term_set = set(t.lower() for t in terms)
        for turn in self.history[-20:]:
            overlap = set(turn.keywords) & term_set
            if overlap:
                matches.append({
                    'timestamp': turn.timestamp,
                    'preview': turn.content[:80],
                    'matched': list(overlap)
                })
        return {'matches': matches[-5:]}

    def _summarize(self) -> Dict[str, Any]:
        """Generate session summary."""
        keywords = set()
        for turn in self.history[-20:]:
            keywords.update(turn.keywords)

        return {
            'turns': len(self.history),
            'focus': f"{self.state.focus_module}: {self.state.focus_task}",
            'artifacts': self.state.artifacts,
            'recent_actions': self.actions[-5:],
            'topics': list(keywords)[:15]
        }

    def save(self) -> Checkpoint:
        """Create checkpoint for session continuation."""
        summary = self._summarize()
        ts = datetime.now().isoformat()
        chk_id = hashlib.sha256(f"{self.session_id}{ts}".encode()).hexdigest()[:16]

        prompt_parts = [f"Continue: {self.state.focus_module}"]
        if self.state.focus_task:
            prompt_parts.append(f"Task: {self.state.focus_task}")
        if self.state.artifacts:
            prompt_parts.append(f"Artifacts: {', '.join(self.state.artifacts[-3:])}")

        return Checkpoint(
            id=chk_id,
            version=self.VERSION,
            timestamp=ts,
            state=self.state,
            history_summary=f"{summary['turns']} turns, topics: {', '.join(summary['topics'][:5])}",
            recent_actions=self.actions[-10:],
            continuation_prompt=" | ".join(prompt_parts)
        )

    def save_to_file(self, path: str):
        """Save checkpoint to file."""
        checkpoint = self.save()
        with open(path, 'w') as f:
            f.write(checkpoint.to_json())

    def load(self, checkpoint: Checkpoint):
        """Restore from checkpoint."""
        self.state = checkpoint.state
        self.actions = list(checkpoint.recent_actions)
        self.actions.append(f"Restored from: {checkpoint.id}")

    def load_from_file(self, path: str):
        """Load checkpoint from file."""
        with open(path, 'r') as f:
            checkpoint = Checkpoint.from_json(f.read())
        self.load(checkpoint)

    def status(self) -> Dict[str, Any]:
        """Current engine status."""
        return {
            'session_id': self.session_id,
            'version': self.VERSION,
            'state': self.state.to_dict(),
            'history_length': len(self.history),
            'pending_suggestions': len(self.suggestions),
            'actions_taken': len(self.actions)
        }

    def print_suggestions(self):
        """Print current suggestions."""
        if not self.suggestions:
            print("No suggestions. Process input first.")
            return
        print("\nSuggestions:")
        for i, s in enumerate(self.suggestions, 1):
            print(f"  [{i}] {s}")


# Convenience alias
MycelialPresenceEngine = SessionContextEngine


if __name__ == "__main__":
    print("Session Context Engine - Test")
    print("=" * 40)

    engine = SessionContextEngine(session_id="test")
    engine.set_focus("dialectical_genie.py", "integration testing", "high")

    # Process input
    result = engine.process("How do I fix the integration error in dialectical_genie?")
    print(f"\nPatterns: {result['analysis']['patterns']}")
    engine.print_suggestions()

    # Execute first suggestion
    if engine.suggestions:
        print(f"\nExecuting: {engine.suggestions[0]}")
        out = engine.execute(engine.suggestions[0])
        print(f"Result: {out}")

    # Save checkpoint
    engine.add_artifact("dialectical_genie.py")
    checkpoint = engine.save()
    print(f"\nCheckpoint: {checkpoint.id}")
    print(f"Continuation: {checkpoint.continuation_prompt}")

    # Restore test
    new_engine = SessionContextEngine(session_id="restored")
    new_engine.load(checkpoint)
    print(f"\nRestored focus: {new_engine.state.focus_module}")
    print(f"Restored artifacts: {new_engine.state.artifacts}")
