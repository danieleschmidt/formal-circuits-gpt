"""Database models for formal verification data."""

import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ProofCache:
    """Cached proof entry."""

    circuit_hash: str
    properties_hash: str
    prover: str
    proof_code: str
    verification_status: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 1
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}

    @classmethod
    def create_hash(cls, circuit_code: str) -> str:
        """Create hash for circuit code."""
        return hashlib.sha256(circuit_code.encode()).hexdigest()

    @classmethod
    def create_properties_hash(cls, properties: List[str]) -> str:
        """Create hash for properties list."""
        properties_str = json.dumps(sorted(properties), sort_keys=True)
        return hashlib.sha256(properties_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "circuit_hash": self.circuit_hash,
            "properties_hash": self.properties_hash,
            "prover": self.prover,
            "proof_code": self.proof_code,
            "verification_status": self.verification_status,
            "access_count": self.access_count,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row) -> "ProofCache":
        """Create from database row."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return cls(
            id=row["id"],
            circuit_hash=row["circuit_hash"],
            properties_hash=row["properties_hash"],
            prover=row["prover"],
            proof_code=row["proof_code"],
            verification_status=row["verification_status"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            access_count=row["access_count"],
            metadata=metadata,
        )


@dataclass
class CircuitModel:
    """Circuit model representation."""

    name: str
    hdl_type: str
    source_code: str
    ast_json: Optional[str] = None
    module_count: int = 0
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}

    def get_hash(self) -> str:
        """Get hash of the circuit source code."""
        return ProofCache.create_hash(self.source_code)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "name": self.name,
            "hdl_type": self.hdl_type,
            "source_code": self.source_code,
            "ast_json": self.ast_json,
            "module_count": self.module_count,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row) -> "CircuitModel":
        """Create from database row."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return cls(
            id=row["id"],
            name=row["name"],
            hdl_type=row["hdl_type"],
            source_code=row["source_code"],
            ast_json=row["ast_json"],
            module_count=row["module_count"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            ),
            metadata=metadata,
        )


@dataclass
class VerificationResult:
    """Verification result record."""

    circuit_id: int
    properties: List[str]
    prover: str
    status: str
    proof_code: Optional[str] = None
    errors: List[str] = None
    execution_time: Optional[float] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.errors = self.errors or []
        self.metadata = self.metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "circuit_id": self.circuit_id,
            "properties": json.dumps(self.properties),
            "prover": self.prover,
            "status": self.status,
            "proof_code": self.proof_code,
            "errors": json.dumps(self.errors),
            "execution_time": self.execution_time,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row) -> "VerificationResult":
        """Create from database row."""
        properties = json.loads(row["properties"]) if row["properties"] else []
        errors = json.loads(row["errors"]) if row["errors"] else []
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return cls(
            id=row["id"],
            circuit_id=row["circuit_id"],
            properties=properties,
            prover=row["prover"],
            status=row["status"],
            proof_code=row["proof_code"],
            errors=errors,
            execution_time=row["execution_time"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            metadata=metadata,
        )


@dataclass
class LemmaCache:
    """Cached lemma for reuse."""

    lemma_hash: str
    lemma_name: str
    statement: str
    proof: str
    prover: str
    usage_count: int = 0
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        self.metadata = self.metadata or {}

    @classmethod
    def create_hash(cls, statement: str, prover: str) -> str:
        """Create hash for lemma statement and prover."""
        content = f"{prover}:{statement}"
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "lemma_hash": self.lemma_hash,
            "lemma_name": self.lemma_name,
            "statement": self.statement,
            "proof": self.proof,
            "prover": self.prover,
            "usage_count": self.usage_count,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_row(cls, row) -> "LemmaCache":
        """Create from database row."""
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}

        return cls(
            id=row["id"],
            lemma_hash=row["lemma_hash"],
            lemma_name=row["lemma_name"],
            statement=row["statement"],
            proof=row["proof"],
            prover=row["prover"],
            usage_count=row["usage_count"],
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            metadata=metadata,
        )
