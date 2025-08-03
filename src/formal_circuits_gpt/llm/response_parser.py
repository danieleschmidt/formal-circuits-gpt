"""Parser for LLM responses containing formal proofs."""

import re
from typing import List, Dict, Any, Optional, Tuple


class ResponseParseError(Exception):
    """Exception raised for response parsing errors."""
    pass


class ResponseParser:
    """Parses LLM responses to extract formal proof code."""
    
    def __init__(self):
        """Initialize response parser with patterns."""
        self.isabelle_patterns = self._init_isabelle_patterns()
        self.coq_patterns = self._init_coq_patterns()
    
    def _init_isabelle_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize Isabelle parsing patterns."""
        return {
            "theory_block": re.compile(
                r"theory\s+\w+.*?end", 
                re.DOTALL | re.IGNORECASE
            ),
            "lemma_block": re.compile(
                r"lemma\s+\w+\s*:.*?(?=lemma|\Z)", 
                re.DOTALL | re.IGNORECASE
            ),
            "proof_block": re.compile(
                r"proof.*?qed", 
                re.DOTALL | re.IGNORECASE
            ),
            "definition_block": re.compile(
                r"definition\s+\w+.*?(?=definition|lemma|theorem|\Z)",
                re.DOTALL | re.IGNORECASE
            )
        }
    
    def _init_coq_patterns(self) -> Dict[str, re.Pattern]:
        """Initialize Coq parsing patterns."""
        return {
            "definition_block": re.compile(
                r"Definition\s+\w+.*?\.(?=\s*(?:Definition|Lemma|Theorem|\Z))",
                re.DOTALL | re.IGNORECASE
            ),
            "lemma_block": re.compile(
                r"Lemma\s+\w+.*?Qed\.",
                re.DOTALL | re.IGNORECASE
            ),
            "theorem_block": re.compile(
                r"Theorem\s+\w+.*?Qed\.",
                re.DOTALL | re.IGNORECASE
            ),
            "proof_block": re.compile(
                r"Proof\..*?Qed\.",
                re.DOTALL | re.IGNORECASE
            )
        }
    
    def extract_proof_code(self, response_content: str, prover: str = "isabelle") -> str:
        """Extract formal proof code from LLM response.
        
        Args:
            response_content: Raw LLM response
            prover: Target prover ("isabelle" or "coq")
            
        Returns:
            Extracted proof code
            
        Raises:
            ResponseParseError: If proof code cannot be extracted
        """
        try:
            # Clean up the response
            cleaned_content = self._clean_response(response_content)
            
            if prover == "isabelle":
                return self._extract_isabelle_code(cleaned_content)
            elif prover == "coq":
                return self._extract_coq_code(cleaned_content)
            else:
                raise ResponseParseError(f"Unsupported prover: {prover}")
                
        except Exception as e:
            raise ResponseParseError(f"Failed to extract proof code: {str(e)}") from e
    
    def _clean_response(self, content: str) -> str:
        """Clean up LLM response content."""
        # Remove markdown code blocks
        content = re.sub(r"```(?:isabelle|coq|hol)?\n?(.*?)\n?```", 
                        r"\1", content, flags=re.DOTALL)
        
        # Remove extra whitespace
        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)
        
        # Remove leading/trailing whitespace
        content = content.strip()
        
        return content
    
    def _extract_isabelle_code(self, content: str) -> str:
        """Extract Isabelle theory code from response."""
        # First try to find complete theory block
        theory_match = self.isabelle_patterns["theory_block"].search(content)
        if theory_match:
            return theory_match.group(0)
        
        # If no complete theory, try to reconstruct from parts
        parts = []
        
        # Look for theory header
        theory_header = re.search(r"theory\s+\w+.*?begin", content, 
                                re.DOTALL | re.IGNORECASE)
        if theory_header:
            parts.append(theory_header.group(0))
        else:
            # Create minimal theory header
            parts.append("theory Generated_Proof\n  imports Main\nbegin")
        
        # Extract definitions
        definitions = self.isabelle_patterns["definition_block"].findall(content)
        parts.extend(definitions)
        
        # Extract lemmas
        lemmas = self.isabelle_patterns["lemma_block"].findall(content)
        parts.extend(lemmas)
        
        # Add theory end
        if not content.strip().endswith("end"):
            parts.append("end")
        
        if len(parts) <= 2:  # Only header and end
            # Fallback: return cleaned content as-is
            return content
        
        return "\n\n".join(parts)
    
    def _extract_coq_code(self, content: str) -> str:
        """Extract Coq vernacular code from response."""
        # Look for Require statements at the beginning
        requires = re.findall(r"Require.*?\.", content, re.IGNORECASE)
        
        # Extract definitions
        definitions = self.coq_patterns["definition_block"].findall(content)
        
        # Extract lemmas and theorems
        lemmas = self.coq_patterns["lemma_block"].findall(content)
        theorems = self.coq_patterns["theorem_block"].findall(content)
        
        # Combine all parts
        parts = requires + definitions + lemmas + theorems
        
        if not parts:
            # Fallback: return cleaned content as-is
            return content
        
        return "\n\n".join(parts)
    
    def extract_errors(self, response_content: str) -> List[str]:
        """Extract error messages from LLM response."""
        errors = []
        
        # Look for common error patterns
        error_patterns = [
            r"Error:?\s*(.+)",
            r"Failed:?\s*(.+)", 
            r"Exception:?\s*(.+)",
            r"Syntax error:?\s*(.+)",
            r"Type error:?\s*(.+)"
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, response_content, re.IGNORECASE)
            errors.extend(matches)
        
        return [error.strip() for error in errors if error.strip()]
    
    def extract_lemmas(self, response_content: str, prover: str = "isabelle") -> List[Dict[str, str]]:
        """Extract individual lemmas from response.
        
        Returns:
            List of dictionaries with 'name', 'statement', and 'proof' keys
        """
        lemmas = []
        
        if prover == "isabelle":
            # Pattern to match lemma name, statement, and proof
            lemma_pattern = re.compile(
                r"lemma\s+(\w+)\s*:\s*\"([^\"]+)\"\s*(.*?)(?=lemma|\Z)",
                re.DOTALL | re.IGNORECASE
            )
            
            for match in lemma_pattern.finditer(response_content):
                lemmas.append({
                    "name": match.group(1),
                    "statement": match.group(2).strip(),
                    "proof": match.group(3).strip()
                })
                
        elif prover == "coq":
            # Pattern for Coq lemmas
            lemma_pattern = re.compile(
                r"Lemma\s+(\w+)\s*:\s*(.*?)\.\s*(Proof\..*?Qed\.)",
                re.DOTALL | re.IGNORECASE
            )
            
            for match in lemma_pattern.finditer(response_content):
                lemmas.append({
                    "name": match.group(1),
                    "statement": match.group(2).strip(),
                    "proof": match.group(3).strip()
                })
        
        return lemmas
    
    def validate_syntax(self, proof_code: str, prover: str = "isabelle") -> Tuple[bool, List[str]]:
        """Basic syntax validation for proof code.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if prover == "isabelle":
            issues.extend(self._validate_isabelle_syntax(proof_code))
        elif prover == "coq":
            issues.extend(self._validate_coq_syntax(proof_code))
        
        return len(issues) == 0, issues
    
    def _validate_isabelle_syntax(self, proof_code: str) -> List[str]:
        """Basic Isabelle syntax validation."""
        issues = []
        
        # Check for theory structure
        if "theory" not in proof_code.lower():
            issues.append("Missing theory declaration")
        
        if "begin" not in proof_code.lower():
            issues.append("Missing theory begin")
        
        if not proof_code.strip().endswith("end"):
            issues.append("Missing theory end")
        
        # Check for balanced quotes
        quote_count = proof_code.count('"')
        if quote_count % 2 != 0:
            issues.append("Unbalanced quotes")
        
        # Check for balanced parentheses
        paren_count = proof_code.count('(') - proof_code.count(')')
        if paren_count != 0:
            issues.append("Unbalanced parentheses")
        
        return issues
    
    def _validate_coq_syntax(self, proof_code: str) -> List[str]:
        """Basic Coq syntax validation."""
        issues = []
        
        # Check that statements end with periods
        lines = [line.strip() for line in proof_code.split('\n') if line.strip()]
        
        for line in lines:
            # Skip comments and empty lines
            if line.startswith('(*') or not line:
                continue
            
            # Check if statement lines end with period
            if any(line.startswith(kw) for kw in ['Definition', 'Lemma', 'Theorem', 'Proof']):
                if not line.endswith('.') and not line.endswith('Qed.'):
                    issues.append(f"Statement may be missing period: {line[:50]}...")
        
        # Check for balanced parentheses
        paren_count = proof_code.count('(') - proof_code.count(')')
        if paren_count != 0:
            issues.append("Unbalanced parentheses")
        
        return issues
    
    def extract_proof_statistics(self, proof_code: str, prover: str = "isabelle") -> Dict[str, int]:
        """Extract statistics about the proof code."""
        stats = {
            "total_lines": len(proof_code.split('\n')),
            "non_empty_lines": len([line for line in proof_code.split('\n') if line.strip()]),
            "lemmas": 0,
            "definitions": 0,
            "theorems": 0
        }
        
        if prover == "isabelle":
            stats["lemmas"] = len(re.findall(r"lemma\s+\w+", proof_code, re.IGNORECASE))
            stats["definitions"] = len(re.findall(r"definition\s+\w+", proof_code, re.IGNORECASE))
            stats["theorems"] = len(re.findall(r"theorem\s+\w+", proof_code, re.IGNORECASE))
        
        elif prover == "coq":
            stats["lemmas"] = len(re.findall(r"Lemma\s+\w+", proof_code, re.IGNORECASE))
            stats["definitions"] = len(re.findall(r"Definition\s+\w+", proof_code, re.IGNORECASE))
            stats["theorems"] = len(re.findall(r"Theorem\s+\w+", proof_code, re.IGNORECASE))
        
        return stats