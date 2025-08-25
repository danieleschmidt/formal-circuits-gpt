"""Tests for core CircuitVerifier functionality."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from formal_circuits_gpt.core import CircuitVerifier, ProofResult, ProverResult
from formal_circuits_gpt.exceptions import VerificationError
from formal_circuits_gpt.parsers.ast_nodes import CircuitAST, Module, Port, SignalType


class TestCircuitVerifier:
    """Test cases for CircuitVerifier class."""

    def test_init_default_params(self):
        """Test CircuitVerifier initialization with default parameters."""
        verifier = CircuitVerifier(strict_mode=False)
        assert verifier.prover == "isabelle"
        assert verifier.model == "gpt-4-turbo"
        assert verifier.temperature == 0.1
        assert verifier.refinement_rounds == 5
        assert verifier.debug_mode is False

    def test_init_custom_params(self):
        """Test CircuitVerifier initialization with custom parameters."""
        verifier = CircuitVerifier(
            prover="coq",
            model="gpt-3.5-turbo",
            temperature=0.5,
            refinement_rounds=10,
            debug_mode=True
        )
        assert verifier.prover == "coq"
        assert verifier.model == "gpt-3.5-turbo"
        assert verifier.temperature == 0.5
        assert verifier.refinement_rounds == 10
        assert verifier.debug_mode is True

    @patch('formal_circuits_gpt.core.CircuitVerifier._parse_hdl')
    @patch('formal_circuits_gpt.core.CircuitVerifier._generate_proof_with_llm')
    @patch('formal_circuits_gpt.core.CircuitVerifier._verify_with_prover')
    def test_verify_basic_success(self, mock_verify, mock_generate, mock_parse):
        """Test basic verification success flow."""
        # Setup mocks
        mock_ast = Mock(spec=CircuitAST)
        mock_ast.modules = []  # Add modules attribute
        mock_parse.return_value = mock_ast
        
        mock_generate.return_value = "proof code"
        
        mock_result = Mock(spec=ProverResult)
        mock_result.success = True
        mock_result.errors = []
        mock_verify.return_value = mock_result
        
        verifier = CircuitVerifier(strict_mode=False)
        
        # Mock the property generator instance
        with patch.object(verifier, 'property_generator') as mock_prop_gen:
            mock_prop_gen.generate_properties.return_value = [
                Mock(formula="sum == a + b")
            ]
            
            result = verifier.verify("module test(); endmodule")
            
            assert result.status == "VERIFIED"
            assert result.proof_code == "proof code"
            assert len(result.errors) == 0

    @patch('formal_circuits_gpt.core.CircuitVerifier._parse_hdl')
    def test_verify_parse_error(self, mock_parse):
        """Test verification with parse error."""
        mock_parse.side_effect = Exception("Parse error")
        
        verifier = CircuitVerifier(strict_mode=False)
        with pytest.raises(VerificationError, match="Failed to parse HDL code: Parse error"):
            verifier.verify("invalid hdl")

    def test_verify_file_not_exists(self):
        """Test verify_file with non-existent file."""
        verifier = CircuitVerifier(strict_mode=False)
        with pytest.raises(VerificationError, match="File not found"):
            verifier.verify_file("nonexistent.v")

    def test_verify_file_success(self):
        """Test verify_file with valid file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write("module test(); endmodule")
            temp_file = f.name
        
        try:
            with patch.object(CircuitVerifier, 'verify') as mock_verify:
                mock_verify.return_value = Mock(spec=ProofResult)
                
                verifier = CircuitVerifier(strict_mode=False)
                verifier.verify_file(temp_file)
                
                mock_verify.assert_called_once_with("module test(); endmodule", None, 3600)
        finally:
            os.unlink(temp_file)

    def test_parse_hdl_verilog(self):
        """Test HDL parsing for Verilog."""
        verifier = CircuitVerifier(strict_mode=False)
        
        with patch.object(verifier.verilog_parser, 'parse') as mock_parse:
            mock_parse.return_value = Mock(spec=CircuitAST)
            
            verilog_code = "module test(); endmodule"
            verifier._parse_hdl(verilog_code)
            
            mock_parse.assert_called_once_with(verilog_code)

    def test_parse_hdl_vhdl(self):
        """Test HDL parsing for VHDL."""
        verifier = CircuitVerifier(strict_mode=False)
        
        with patch.object(verifier.vhdl_parser, 'parse') as mock_parse:
            mock_parse.return_value = Mock(spec=CircuitAST)
            
            vhdl_code = "entity test is end test;"
            verifier._parse_hdl(vhdl_code)
            
            mock_parse.assert_called_once_with(vhdl_code)


class TestProofResult:
    """Test cases for ProofResult class."""

    def test_init_basic(self):
        """Test ProofResult initialization."""
        result = ProofResult("VERIFIED", "proof code")
        assert result.status == "VERIFIED"
        assert result.proof_code == "proof code"
        assert result.errors == []
        assert result.properties_verified == []

    def test_init_with_all_params(self):
        """Test ProofResult with all parameters."""
        mock_ast = Mock(spec=CircuitAST)
        result = ProofResult(
            status="FAILED",
            proof_code="partial proof",
            errors=["error 1", "error 2"],
            properties_verified=["prop1", "prop2"],
            ast=mock_ast
        )
        
        assert result.status == "FAILED"
        assert result.proof_code == "partial proof"
        assert result.errors == ["error 1", "error 2"]
        assert result.properties_verified == ["prop1", "prop2"]
        assert result.ast == mock_ast

    def test_isabelle_code_property(self):
        """Test isabelle_code property."""
        result = ProofResult("VERIFIED", "theory Test begin lemma test: True proof auto qed end")
        assert "theory" in result.isabelle_code.lower()
        
        result_no_isabelle = ProofResult("VERIFIED", "some other proof")
        assert result_no_isabelle.isabelle_code == ""

    def test_coq_code_property(self):
        """Test coq_code property."""
        result = ProofResult("VERIFIED", "Require Import Nat. Definition test := 1.")
        assert "Require" in result.coq_code
        
        result_definition = ProofResult("VERIFIED", "Definition test := 1.")
        assert "Definition" in result_definition.coq_code

    def test_export_latex(self):
        """Test LaTeX export functionality."""
        result = ProofResult(
            status="VERIFIED",
            proof_code="proof code",
            properties_verified=["prop1", "prop2"]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as f:
            temp_file = f.name
        
        try:
            result.export_latex(temp_file)
            
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "VERIFIED" in content
                assert "prop1" in content
                assert "prop2" in content
                assert "proof code" in content
        finally:
            os.unlink(temp_file)

    def test_export_systemverilog_assertions(self):
        """Test SystemVerilog assertions export."""
        result = ProofResult(
            status="VERIFIED",
            properties_verified=["a + b == sum", "count >= 0"]
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sv', delete=False) as f:
            temp_file = f.name
        
        try:
            result.export_systemverilog_assertions(temp_file)
            
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "property prop_1" in content
                assert "property prop_2" in content
                assert "assert property" in content
        finally:
            os.unlink(temp_file)


class TestProverResult:
    """Test cases for ProverResult class."""

    def test_init_success(self):
        """Test ProverResult for success case."""
        result = ProverResult(success=True)
        assert result.success is True
        assert result.errors == []
        assert result.output == ""

    def test_init_failure(self):
        """Test ProverResult for failure case."""
        result = ProverResult(
            success=False,
            errors=["syntax error", "type error"],
            output="error output"
        )
        assert result.success is False
        assert result.errors == ["syntax error", "type error"]
        assert result.output == "error output"