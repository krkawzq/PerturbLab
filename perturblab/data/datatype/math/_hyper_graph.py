from abc import ABC, abstractmethod
from typing import Hashable, Any, Tuple, Set, TypeVar, Generic, Type

class Entity(ABC):
    @abstractmethod
    def unique_id(self) -> Hashable:
        """Return a hashable identifier for this entity."""
        pass

    def __hash__(self):
        return hash(self.unique_id())

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.unique_id() == other.unique_id()

    def __repr__(self):
        return f"<{self.__class__.__name__}:{self.unique_id()}>"


class Atom(Entity, ABC):
    """
    Atom: An indivisible relationship unit. Also acts as an Entity.
    """
    @property
    @abstractmethod
    def constituents(self) -> Tuple[Entity, ...]:
        """
        Entities involved in this atom.
        For undirected relationships, tuple ordering should be consistent to ensure uniqueness.
        """
        pass

    @property
    @abstractmethod
    def relation(self) -> Any:
        """Payload for the relationship: weight, type label, confidence, etc."""
        pass

    def unique_id(self) -> Hashable:
        # By default: (ClassName, constituent ids)
        # Entities in the same relationship class and with same constituents are unique.
        # Override this if you want to support multigraphs (multiple edges).
        const_ids = tuple(e.unique_id() for e in self.constituents)
        return (self.__class__.__name__,) + const_ids


# ==========================================
# 3. Relationship container - type safe generic
# ==========================================

# Type variable, constrained to Atom subclasses
T_Atom = TypeVar("T_Atom", bound=Atom)

class Relationship(ABC, Generic[T_Atom]):
    """
    Relationship: A container for a set of Atoms of a specific type.
    The current domain (all involved entities) is calculated dynamically.
    """
    def __init__(self, name: str, atom_cls: Type[T_Atom]):
        self.name = name
        self._bound_class = atom_cls
        self._atoms: Set[T_Atom] = set()   # Store core atoms

    @property
    def domain(self) -> Set[Entity]:
        """Return all entities covered by the relationship (computed each time)."""
        entities = set()
        for atom in self._atoms:
            entities.update(atom.constituents)
        return entities

    @property
    def atoms(self) -> Set[T_Atom]:
        """Return the set of atoms."""
        return self._atoms

    def add(self, atom: T_Atom):
        """
        Register an atom. No internal domain maintenance.
        """
        # Runtime type check for strong typing
        if not isinstance(atom, self._bound_class):
            raise TypeError(
                f"Relationship '{self.name}' expects atom type {self._bound_class.__name__}, "
                f"but received {type(atom).__name__}."
            )
        self._atoms.add(atom)  # Set ensures uniqueness

    def __len__(self):
        return len(self._atoms)

    def __contains__(self, atom: T_Atom):
        return atom in self._atoms

    def __repr__(self):
        return f"Relationship(name='{self.name}', atoms={len(self)}, domain={len(self.domain)})"

# ==========================================
# 4. Example implementation
# ==========================================

# Example: undirected gene coexpression atom
class Gene(Entity):
    def __init__(self, symbol):
        self.symbol = symbol
    def unique_id(self):
        return self.symbol

class CoexpAtom(Atom):
    def __init__(self, g1: Gene, g2: Gene, score: float):
        # Order matters for undirected case: (A, B) == (B, A)
        self._constituents = tuple(sorted([g1, g2], key=lambda x: str(x.unique_id())))
        self._score = score

    @property
    def constituents(self):
        return self._constituents

    @property
    def relation(self):
        return self._score
