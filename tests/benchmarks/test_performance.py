"""Performance benchmark tests."""

import time
import pytest
from pathlib import Path
from tests.fixtures import ALL_FIXTURES, SIMPLE_FIXTURES, COMPLEX_FIXTURES


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for various operations."""

    def test_parser_performance(self):
        """Benchmark parser performance."""
        try:
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser()
            
            # Time parsing of different circuit sizes
            times = []
            for fixture in SIMPLE_FIXTURES:
                start_time = time.time()
                try:
                    parser.parse(fixture.verilog_code)
                except NotImplementedError:
                    pytest.skip("Parser not implemented yet")
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Simple circuits should parse quickly (< 1 second each)
            for parse_time in times:
                assert parse_time < 1.0, f"Parser took {parse_time:.2f}s, too slow"
            
            avg_time = sum(times) / len(times)
            print(f"Average parse time: {avg_time:.3f}s")
            
        except ImportError:
            pytest.skip("Parser module not available")

    def test_property_inference_performance(self):
        """Benchmark property inference performance."""
        try:
            from formal_circuits_gpt.properties import PropertyInference
            
            inferrer = PropertyInference()
            
            times = []
            for fixture in SIMPLE_FIXTURES:
                start_time = time.time()
                try:
                    inferrer.infer_properties(fixture.verilog_code)
                except NotImplementedError:
                    pytest.skip("Property inference not implemented yet")
                end_time = time.time()
                times.append(end_time - start_time)
            
            # Property inference should be fast (< 5 seconds each)
            for infer_time in times:
                assert infer_time < 5.0, f"Property inference took {infer_time:.2f}s, too slow"
            
            avg_time = sum(times) / len(times)
            print(f"Average property inference time: {avg_time:.3f}s")
            
        except ImportError:
            pytest.skip("Property inference module not available")

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with circuit complexity."""
        try:
            import psutil
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser()
            process = psutil.Process()
            
            initial_memory = process.memory_info().rss
            memory_usage = []
            
            for fixture in ALL_FIXTURES:
                try:
                    parser.parse(fixture.verilog_code)
                    current_memory = process.memory_info().rss
                    memory_increase = current_memory - initial_memory
                    memory_usage.append(memory_increase)
                except NotImplementedError:
                    pytest.skip("Parser not implemented yet")
            
            # Memory usage should not grow excessively
            max_memory_increase = max(memory_usage)
            assert max_memory_increase < 100 * 1024 * 1024, f"Memory usage too high: {max_memory_increase / 1024 / 1024:.1f}MB"
            
            print(f"Max memory increase: {max_memory_increase / 1024 / 1024:.1f}MB")
            
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")

    @pytest.mark.slow
    def test_large_circuit_performance(self):
        """Test performance with larger circuits."""
        # Generate a larger circuit for testing
        large_circuit = self._generate_large_circuit(width=32, depth=8)
        
        try:
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser()
            
            start_time = time.time()
            try:
                parser.parse(large_circuit)
            except NotImplementedError:
                pytest.skip("Parser not implemented yet")
            end_time = time.time()
            
            parse_time = end_time - start_time
            
            # Large circuits should still parse in reasonable time (< 30 seconds)
            assert parse_time < 30.0, f"Large circuit parsing took {parse_time:.2f}s, too slow"
            
            print(f"Large circuit parse time: {parse_time:.3f}s")
            
        except ImportError:
            pytest.skip("Parser module not available")

    def test_concurrent_parsing_performance(self):
        """Test performance of concurrent parsing operations."""
        try:
            import concurrent.futures
            from formal_circuits_gpt.parsers import VerilogParser
            
            def parse_circuit(fixture):
                parser = VerilogParser()
                start_time = time.time()
                try:
                    parser.parse(fixture.verilog_code)
                except NotImplementedError:
                    return None
                end_time = time.time()
                return end_time - start_time
            
            # Test concurrent parsing
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(parse_circuit, fixture) for fixture in SIMPLE_FIXTURES]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            end_time = time.time()
            
            if any(result is None for result in results):
                pytest.skip("Parser not implemented yet")
            
            concurrent_time = end_time - start_time
            sequential_time = sum(results)
            
            # Concurrent parsing should show some speedup
            speedup = sequential_time / concurrent_time
            print(f"Concurrent speedup: {speedup:.2f}x")
            
            # Should have at least some speedup (> 1.2x)
            assert speedup > 1.2, f"Insufficient concurrent speedup: {speedup:.2f}x"
            
        except ImportError:
            pytest.skip("Parser module or concurrent.futures not available")

    def _generate_large_circuit(self, width: int = 32, depth: int = 8) -> str:
        """Generate a large circuit for performance testing."""
        return f"""
        module large_test_circuit(
            input [{width-1}:0] data_in,
            input clk,
            input reset,
            output [{width-1}:0] data_out
        );
            
            {''.join([
                f"reg [{width-1}:0] stage_{i};\n" 
                for i in range(depth)
            ])}
            
            always @(posedge clk) begin
                if (reset) begin
                    {''.join([
                        f"stage_{i} <= {width}'b0;\n                    "
                        for i in range(depth)
                    ])}
                end else begin
                    stage_0 <= data_in;
                    {''.join([
                        f"stage_{i+1} <= stage_{i} + 1;\n                    "
                        for i in range(depth-1)
                    ])}
                end
            end
            
            assign data_out = stage_{depth-1};
            
        endmodule
        """


@pytest.mark.benchmark
class TestThroughputBenchmarks:
    """Throughput benchmarks for batch operations."""

    def test_batch_parsing_throughput(self):
        """Test throughput of batch parsing operations."""
        try:
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser() 
            circuits = [fixture.verilog_code for fixture in ALL_FIXTURES]
            
            start_time = time.time()
            successful_parses = 0
            
            for circuit in circuits:
                try:
                    result = parser.parse(circuit)
                    if result is not None:
                        successful_parses += 1
                except NotImplementedError:
                    pytest.skip("Parser not implemented yet")
                except Exception:
                    # Count parsing failures but continue
                    pass
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if successful_parses == 0:
                pytest.skip("No successful parses to measure throughput")
            
            throughput = successful_parses / total_time
            print(f"Parsing throughput: {throughput:.2f} circuits/second")
            
            # Should achieve reasonable throughput (> 1 circuit/second)
            assert throughput > 1.0, f"Parsing throughput too low: {throughput:.2f} circuits/s"
            
        except ImportError:
            pytest.skip("Parser module not available")

    def test_property_generation_throughput(self):
        """Test throughput of property generation."""
        try:
            from formal_circuits_gpt.properties import PropertyInference
            
            inferrer = PropertyInference()
            circuits = [fixture.verilog_code for fixture in SIMPLE_FIXTURES]
            
            start_time = time.time()
            successful_generations = 0
            
            for circuit in circuits:
                try:
                    properties = inferrer.infer_properties(circuit)
                    if properties and len(properties) > 0:
                        successful_generations += 1
                except NotImplementedError:
                    pytest.skip("Property inference not implemented yet")
                except Exception:
                    # Count failures but continue
                    pass
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if successful_generations == 0:
                pytest.skip("No successful property generations to measure throughput")
            
            throughput = successful_generations / total_time
            print(f"Property generation throughput: {throughput:.2f} circuits/second")
            
            # Should achieve reasonable throughput
            assert throughput > 0.5, f"Property generation throughput too low: {throughput:.2f} circuits/s"
            
        except ImportError:
            pytest.skip("Property inference module not available")


@pytest.mark.benchmark
@pytest.mark.slow
class TestScalabilityBenchmarks:
    """Scalability benchmarks for large-scale operations."""

    def test_circuit_size_scaling(self):
        """Test how performance scales with circuit size."""
        sizes = [8, 16, 32, 64]  # Bit widths to test
        times = []
        
        try:
            from formal_circuits_gpt.parsers import VerilogParser
            parser = VerilogParser()
            
            for size in sizes:
                circuit = self._generate_adder_circuit(size)
                
                start_time = time.time()
                try:
                    parser.parse(circuit)
                except NotImplementedError:
                    pytest.skip("Parser not implemented yet")
                end_time = time.time()
                
                parse_time = end_time - start_time
                times.append(parse_time)
                print(f"{size}-bit circuit: {parse_time:.3f}s")
            
            # Performance should scale reasonably (not exponentially)
            # Check that 64-bit is not more than 10x slower than 8-bit
            if len(times) >= 2:
                scaling_factor = times[-1] / times[0]
                assert scaling_factor < 10.0, f"Performance scaling too poor: {scaling_factor:.2f}x"
                
        except ImportError:
            pytest.skip("Parser module not available")

    def test_property_count_scaling(self):
        """Test how performance scales with number of properties."""
        property_counts = [1, 5, 10, 20]
        times = []
        
        try:
            from formal_circuits_gpt.properties import PropertyValidator
            validator = PropertyValidator()
            
            base_properties = [
                "a >= 0",
                "b >= 0", 
                "sum >= a",
                "sum >= b",
                "sum == a + b"
            ]
            
            for count in property_counts:
                # Create property list by repeating and modifying base properties
                properties = []
                for i in range(count):
                    prop_index = i % len(base_properties)
                    prop = base_properties[prop_index]
                    if i >= len(base_properties):
                        prop = f"({prop}) && true"  # Slightly modify
                    properties.append(prop)
                
                start_time = time.time()
                try:
                    validator.validate_properties(properties)
                except NotImplementedError:
                    pytest.skip("Property validation not implemented yet")
                end_time = time.time()
                
                validation_time = end_time - start_time
                times.append(validation_time)
                print(f"{count} properties: {validation_time:.3f}s")
            
            # Should scale linearly or sub-linearly with property count
            if len(times) >= 2:
                scaling_factor = times[-1] / times[0]
                expected_factor = property_counts[-1] / property_counts[0]
                assert scaling_factor <= expected_factor * 2, f"Property validation scaling too poor: {scaling_factor:.2f}x for {expected_factor}x more properties"
                
        except ImportError:
            pytest.skip("Property validation module not available")

    def _generate_adder_circuit(self, width: int) -> str:
        """Generate an adder circuit of specified width."""
        return f"""
        module adder_{width}bit(
            input [{width-1}:0] a,
            input [{width-1}:0] b,
            input cin,
            output [{width}:0] sum
        );
            assign sum = a + b + cin;
        endmodule
        """


@pytest.mark.benchmark
class TestResourceUsageBenchmarks:
    """Resource usage benchmarks."""

    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency during operations."""
        try:
            import psutil
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser()
            process = psutil.Process()
            
            # Monitor CPU usage during parsing
            cpu_before = process.cpu_percent()
            
            start_time = time.time()
            for fixture in SIMPLE_FIXTURES:
                try:
                    parser.parse(fixture.verilog_code)
                except NotImplementedError:
                    pytest.skip("Parser not implemented yet")
            end_time = time.time()
            
            # Let CPU usage settle
            time.sleep(0.1)
            cpu_after = process.cpu_percent()
            
            duration = end_time - start_time
            print(f"CPU usage during parsing: {cpu_after:.1f}%")
            print(f"Duration: {duration:.3f}s")
            
            # CPU usage should be reasonable (not constantly at 100%)
            assert cpu_after < 95.0, f"CPU usage too high: {cpu_after:.1f}%"
            
        except ImportError:
            pytest.skip("psutil not available for CPU monitoring")

    def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        try:
            import psutil
            import gc
            from formal_circuits_gpt.parsers import VerilogParser
            
            parser = VerilogParser()
            process = psutil.Process()
            
            # Baseline memory usage
            gc.collect()
            initial_memory = process.memory_info().rss
            
            # Perform operations multiple times
            for iteration in range(10):
                for fixture in SIMPLE_FIXTURES:
                    try:
                        parser.parse(fixture.verilog_code)
                    except NotImplementedError:
                        pytest.skip("Parser not implemented yet")
                
                # Force garbage collection
                gc.collect()
                
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                print(f"Iteration {iteration + 1}: Memory increase: {memory_increase / 1024 / 1024:.1f}MB")
            
            # Final memory check
            final_memory = process.memory_info().rss
            total_increase = final_memory - initial_memory
            
            # Memory increase should be bounded (< 50MB for repeated operations)
            assert total_increase < 50 * 1024 * 1024, f"Potential memory leak: {total_increase / 1024 / 1024:.1f}MB increase"
            
        except ImportError:
            pytest.skip("psutil not available for memory monitoring")