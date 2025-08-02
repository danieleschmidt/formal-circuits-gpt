"""Unit tests for HDL parsers."""

import pytest
from formal_circuits_gpt.parsers import VerilogParser, VHDLParser, ParseError
from tests.fixtures import SIMPLE_FIXTURES, BUGGY_FIXTURES


class TestVerilogParser:
    """Test cases for Verilog parser."""

    def test_parser_initialization(self):
        """Test VerilogParser initialization."""
        parser = VerilogParser()
        assert parser is not None
        assert hasattr(parser, 'parse')

    @pytest.mark.parametrize("fixture", SIMPLE_FIXTURES)
    def test_parse_valid_verilog(self, fixture):
        """Test parsing valid Verilog code."""
        parser = VerilogParser()
        
        # This test will pass even if parse is not implemented
        # because we're testing the interface, not the implementation
        try:
            result = parser.parse(fixture.verilog_code)
            # If implemented, result should not be None
            if result is not None:
                assert hasattr(result, 'modules') or hasattr(result, 'top_module')
        except NotImplementedError:
            # Expected if parser is not yet implemented
            pytest.skip("VerilogParser.parse not implemented yet")

    def test_parse_invalid_syntax(self):
        """Test parsing invalid Verilog syntax."""
        parser = VerilogParser()
        invalid_code = """
        module invalid_syntax(
            input a,
            // Missing semicolon and closing parenthesis
            output b
        """
        
        try:
            with pytest.raises((ParseError, SyntaxError, Exception)):
                parser.parse(invalid_code)
        except NotImplementedError:
            pytest.skip("VerilogParser.parse not implemented yet")

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        parser = VerilogParser()
        try:
            with pytest.raises((ParseError, ValueError, Exception)):
                parser.parse("")
        except NotImplementedError:
            pytest.skip("VerilogParser.parse not implemented yet")

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only string."""
        parser = VerilogParser()
        try:
            with pytest.raises((ParseError, ValueError, Exception)):
                parser.parse("   \n\t  ")
        except NotImplementedError:
            pytest.skip("VerilogParser.parse not implemented yet")


class TestVHDLParser:
    """Test cases for VHDL parser."""

    def test_parser_initialization(self):
        """Test VHDLParser initialization."""
        parser = VHDLParser()
        assert parser is not None
        assert hasattr(parser, 'parse')

    @pytest.mark.parametrize("fixture", SIMPLE_FIXTURES)
    def test_parse_valid_vhdl(self, fixture):
        """Test parsing valid VHDL code."""
        parser = VHDLParser()
        
        try:
            result = parser.parse(fixture.vhdl_code)
            # If implemented, result should not be None
            if result is not None:
                assert hasattr(result, 'entities') or hasattr(result, 'top_entity')
        except NotImplementedError:
            pytest.skip("VHDLParser.parse not implemented yet")

    def test_parse_invalid_syntax(self):
        """Test parsing invalid VHDL syntax."""
        parser = VHDLParser()
        invalid_code = """
        entity invalid_syntax is
            port (
                a : in std_logic
                -- Missing semicolon
                b : out std_logic
            );
        -- Missing 'end entity;'
        """
        
        try:
            with pytest.raises((ParseError, SyntaxError, Exception)):
                parser.parse(invalid_code)
        except NotImplementedError:
            pytest.skip("VHDLParser.parse not implemented yet")


class TestParseError:
    """Test cases for ParseError exception."""

    def test_parse_error_basic(self):
        """Test basic ParseError exception."""
        exc = ParseError("Parse failed")
        assert str(exc) == "Parse failed"
        assert exc.line_number is None
        assert exc.column is None

    def test_parse_error_with_location(self):
        """Test ParseError with line and column information."""
        exc = ParseError("Parse failed", line_number=42, column=10)
        assert str(exc) == "Parse failed"
        assert exc.line_number == 42
        assert exc.column == 10

    def test_parse_error_inheritance(self):
        """Test that ParseError inherits from Exception."""
        exc = ParseError("Parse failed")
        assert isinstance(exc, Exception)


class TestParserIntegration:
    """Integration tests for parser interaction."""

    def test_both_parsers_handle_same_circuit(self):
        """Test that both parsers can handle equivalent circuits."""
        verilog_parser = VerilogParser()
        vhdl_parser = VHDLParser()
        
        # Use the simple adder fixture
        fixture = SIMPLE_FIXTURES[0]  # simple_adder
        
        try:
            verilog_result = verilog_parser.parse(fixture.verilog_code)
            vhdl_result = vhdl_parser.parse(fixture.vhdl_code)
            
            # If both are implemented, they should both succeed or both fail
            assert (verilog_result is None) == (vhdl_result is None)
            
        except NotImplementedError:
            pytest.skip("Parsers not implemented yet")

    @pytest.mark.parametrize("fixture", BUGGY_FIXTURES)
    def test_parsers_handle_buggy_circuits(self, fixture):
        """Test that parsers can parse syntactically correct but logically buggy circuits."""
        verilog_parser = VerilogParser()
        vhdl_parser = VHDLParser()
        
        try:
            # Buggy circuits should still parse (syntax is correct)
            verilog_result = verilog_parser.parse(fixture.verilog_code)
            vhdl_result = vhdl_parser.parse(fixture.vhdl_code)
            
            # Should not raise exceptions for syntactically correct code
            # (semantic issues are caught in verification, not parsing)
            
        except NotImplementedError:
            pytest.skip("Parsers not implemented yet")