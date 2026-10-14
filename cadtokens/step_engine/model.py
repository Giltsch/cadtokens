"""
STEP Engine - Data Model

Pure data types for representing ISO 10303-21 (Part 21) files.
Zero external dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


# ============================================================================
# Parameter Types
# ============================================================================

@dataclass(frozen=True)
class Ref:
    """Entity reference (#ID)."""
    entity_id: int

    def __repr__(self):
        return f"#{self.entity_id}"


@dataclass(frozen=True)
class EnumValue:
    """Enumeration value (.VALUE.)."""
    value: str

    def __repr__(self):
        return f".{self.value}."


class Unset:
    """Unset/missing parameter ($). Singleton."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "$"

    def __eq__(self, other):
        return isinstance(other, Unset)

    def __hash__(self):
        return hash("$")


class Derived:
    """Derived/redefined parameter (*). Singleton."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "*"

    def __eq__(self, other):
        return isinstance(other, Derived)

    def __hash__(self):
        return hash("*")


UNSET = Unset()
DERIVED = Derived()


@dataclass
class TypedValue:
    """Typed parameter, e.g. LENGTH_MEASURE(1.0)."""
    type_name: str
    value: 'StepParam'

    def __repr__(self):
        return f"{self.type_name}({self.value})"


# Union type for any STEP parameter
StepParam = Union[int, float, str, Ref, EnumValue, TypedValue, list, Unset, Derived]


# ============================================================================
# Entity Types
# ============================================================================

@dataclass
class EntityPart:
    """One part of an entity (type name + parameters).

    Used for simple entities (single part) and as components of complex entities.
    """
    type_name: str
    params: List[StepParam] = field(default_factory=list)

    def __repr__(self):
        pstr = ", ".join(repr(p) for p in self.params[:3])
        if len(self.params) > 3:
            pstr += ", ..."
        return f"{self.type_name}({pstr})"


@dataclass
class Entity:
    """A STEP entity definition.

    Simple:  #1 = CARTESIAN_POINT('', (0., 0., 0.));
    Complex: #42 = ( GEOMETRIC_REPRESENTATION_CONTEXT(2)
                     PARAMETRIC_REPRESENTATION_CONTEXT() ... );
    """
    entity_id: int
    parts: List[EntityPart]

    @property
    def is_complex(self) -> bool:
        return len(self.parts) > 1

    @property
    def type_name(self) -> str:
        """Primary type name (first part for complex entities)."""
        return self.parts[0].type_name

    @property
    def type_names(self) -> List[str]:
        """All type names (useful for complex entities)."""
        return [p.type_name for p in self.parts]

    @property
    def params(self) -> List[StepParam]:
        """Parameters of the first (or only) part."""
        return self.parts[0].params

    def has_type(self, name: str) -> bool:
        """Check if entity contains this type (checks all parts for complex)."""
        return any(p.type_name == name for p in self.parts)

    def get_part(self, name: str) -> Optional[EntityPart]:
        """Get a specific part by type name."""
        for p in self.parts:
            if p.type_name == name:
                return p
        return None

    def __repr__(self):
        if self.is_complex:
            names = " ".join(f"{p.type_name}(...)" for p in self.parts)
            return f"#{self.entity_id} = ({names})"
        return f"#{self.entity_id} = {self.parts[0]}"


# ============================================================================
# File-level Structures
# ============================================================================

@dataclass
class HeaderSection:
    """STEP file header."""
    file_description: Optional[str] = None
    file_name: Optional[str] = None
    file_schema: Optional[List[str]] = None
    raw: str = ""


@dataclass
class StepFile:
    """Complete parsed STEP file."""
    header: HeaderSection
    entities: Dict[int, Entity] = field(default_factory=dict)
    _max_id: int = 0

    def __post_init__(self):
        if self.entities:
            self._max_id = max(self.entities.keys())

    @property
    def next_id(self) -> int:
        """Next available entity ID."""
        return self._max_id + 1

    def __len__(self) -> int:
        return len(self.entities)

    def __getitem__(self, entity_id: int) -> Entity:
        return self.entities[entity_id]

    def __contains__(self, entity_id: int) -> bool:
        return entity_id in self.entities

    def __iter__(self):
        return iter(self.entities.values())

    def get(self, entity_id: int, default=None) -> Optional[Entity]:
        return self.entities.get(entity_id, default)
