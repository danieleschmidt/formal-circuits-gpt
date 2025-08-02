"""Test fixture circuits for formal-circuits-gpt testing."""

from typing import Dict, List, NamedTuple


class CircuitFixture(NamedTuple):
    """A test circuit with expected properties and verification outcomes."""
    name: str
    verilog_code: str
    vhdl_code: str
    properties: List[str]
    should_verify: bool
    complexity: str  # 'simple', 'medium', 'complex'
    circuit_type: str  # 'combinational', 'sequential', 'mixed'
    description: str


# Simple Combinational Circuits
SIMPLE_ADDER = CircuitFixture(
    name="simple_adder",
    verilog_code="""
module simple_adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    assign sum = a + b;
endmodule
""",
    vhdl_code="""
entity simple_adder is
    port (
        a : in std_logic_vector(3 downto 0);
        b : in std_logic_vector(3 downto 0);
        sum : out std_logic_vector(4 downto 0)
    );
end entity;

architecture behavioral of simple_adder is
begin
    sum <= std_logic_vector(unsigned('0' & a) + unsigned('0' & b));
end architecture;
""",
    properties=[
        "sum == a + b",
        "sum >= a",
        "sum >= b",
        "sum <= 30"  # 4-bit inputs max
    ],
    should_verify=True,
    complexity="simple",
    circuit_type="combinational",
    description="Basic 4-bit adder with carry out"
)

MULTIPLEXER_2TO1 = CircuitFixture(
    name="mux_2to1",
    verilog_code="""
module mux_2to1(
    input [7:0] in0,
    input [7:0] in1,
    input sel,
    output [7:0] out
);
    assign out = sel ? in1 : in0;
endmodule
""",
    vhdl_code="""
entity mux_2to1 is
    port (
        in0 : in std_logic_vector(7 downto 0);
        in1 : in std_logic_vector(7 downto 0);
        sel : in std_logic;
        out : out std_logic_vector(7 downto 0)
    );
end entity;

architecture behavioral of mux_2to1 is
begin
    out <= in1 when sel = '1' else in0;
end architecture;
""",
    properties=[
        "sel == 0 -> out == in0",
        "sel == 1 -> out == in1",
        "out == in0 || out == in1"
    ],
    should_verify=True,
    complexity="simple",
    circuit_type="combinational",
    description="2-to-1 multiplexer with 8-bit data width"
)

DECODER_2TO4 = CircuitFixture(
    name="decoder_2to4",
    verilog_code="""
module decoder_2to4(
    input [1:0] sel,
    input enable,
    output [3:0] out
);
    assign out = enable ? (4'b0001 << sel) : 4'b0000;
endmodule
""",
    vhdl_code="""
entity decoder_2to4 is
    port (
        sel : in std_logic_vector(1 downto 0);
        enable : in std_logic;
        out : out std_logic_vector(3 downto 0)
    );
end entity;

architecture behavioral of decoder_2to4 is
begin
    process(sel, enable)
    begin
        if enable = '1' then
            case sel is
                when "00" => out <= "0001";
                when "01" => out <= "0010";
                when "10" => out <= "0100";
                when "11" => out <= "1000";
                when others => out <= "0000";
            end case;
        else
            out <= "0000";
        end if;
    end process;
end architecture;
""",
    properties=[
        "enable == 0 -> out == 0",
        "enable == 1 && sel == 0 -> out == 1",
        "enable == 1 && sel == 1 -> out == 2",
        "enable == 1 && sel == 2 -> out == 4",
        "enable == 1 && sel == 3 -> out == 8",
        "count_ones(out) <= 1"  # At most one output active
    ],
    should_verify=True,
    complexity="simple",
    circuit_type="combinational",
    description="2-to-4 decoder with enable signal"
)

# Medium Complexity Circuits
COUNTER_8BIT = CircuitFixture(
    name="counter_8bit",
    verilog_code="""
module counter_8bit(
    input clk,
    input reset,
    input enable,
    output reg [7:0] count
);
    always @(posedge clk) begin
        if (reset)
            count <= 8'b0;
        else if (enable)
            count <= count + 1;
    end
endmodule
""",
    vhdl_code="""
entity counter_8bit is
    port (
        clk : in std_logic;
        reset : in std_logic;
        enable : in std_logic;
        count : out std_logic_vector(7 downto 0)
    );
end entity;

architecture behavioral of counter_8bit is
    signal count_reg : unsigned(7 downto 0);
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                count_reg <= (others => '0');
            elsif enable = '1' then
                count_reg <= count_reg + 1;
            end if;
        end if;
    end process;
    
    count <= std_logic_vector(count_reg);
end architecture;
""",
    properties=[
        "reset -> next(count) == 0",
        "!reset && enable && count < 255 -> next(count) == count + 1",
        "!reset && enable && count == 255 -> next(count) == 0",  # Overflow wrap
        "!reset && !enable -> next(count) == count",  # Hold value
        "count >= 0 && count <= 255"  # Range check
    ],
    should_verify=True,
    complexity="medium",
    circuit_type="sequential",
    description="8-bit counter with reset and enable"
)

SHIFT_REGISTER = CircuitFixture(
    name="shift_register",
    verilog_code="""
module shift_register(
    input clk,
    input reset,
    input shift_enable,
    input serial_in,
    output serial_out,
    output [7:0] parallel_out
);
    reg [7:0] shift_reg;
    
    always @(posedge clk) begin
        if (reset)
            shift_reg <= 8'b0;
        else if (shift_enable)
            shift_reg <= {shift_reg[6:0], serial_in};
    end
    
    assign serial_out = shift_reg[7];
    assign parallel_out = shift_reg;
endmodule
""",
    vhdl_code="""
entity shift_register is
    port (
        clk : in std_logic;
        reset : in std_logic;
        shift_enable : in std_logic;
        serial_in : in std_logic;
        serial_out : out std_logic;
        parallel_out : out std_logic_vector(7 downto 0)
    );
end entity;

architecture behavioral of shift_register is
    signal shift_reg : std_logic_vector(7 downto 0);
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                shift_reg <= (others => '0');
            elsif shift_enable = '1' then
                shift_reg <= shift_reg(6 downto 0) & serial_in;
            end if;
        end if;
    end process;
    
    serial_out <= shift_reg(7);
    parallel_out <= shift_reg;
end architecture;
""",
    properties=[
        "reset -> next(parallel_out) == 0",
        "shift_enable -> next(serial_out) == parallel_out[6]",
        "shift_enable -> next(parallel_out[0]) == serial_in",
        "!shift_enable && !reset -> next(parallel_out) == parallel_out"
    ],
    should_verify=True,
    complexity="medium",
    circuit_type="sequential",
    description="8-bit left shift register with parallel output"
)

# Complex Circuits
FSM_TRAFFIC_LIGHT = CircuitFixture(
    name="traffic_light_fsm",
    verilog_code="""
module traffic_light_fsm(
    input clk,
    input reset,
    input timer_expired,
    output reg [1:0] state,
    output reg [2:0] lights  // [red, yellow, green]
);
    parameter IDLE = 2'b00, GREEN = 2'b01, YELLOW = 2'b10, RED = 2'b11;
    
    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
        end else begin
            case (state)
                IDLE:   if (timer_expired) state <= GREEN;
                GREEN:  if (timer_expired) state <= YELLOW;
                YELLOW: if (timer_expired) state <= RED;
                RED:    if (timer_expired) state <= GREEN;
                default: state <= IDLE;
            endcase
        end
    end
    
    always @(*) begin
        case (state)
            IDLE:   lights = 3'b100;  // Red only
            GREEN:  lights = 3'b001;  // Green only
            YELLOW: lights = 3'b010;  // Yellow only
            RED:    lights = 3'b100;  // Red only
            default: lights = 3'b000;
        endcase
    end
endmodule
""",
    vhdl_code="""
type state_type is (IDLE, GREEN, YELLOW, RED);

entity traffic_light_fsm is
    port (
        clk : in std_logic;
        reset : in std_logic;
        timer_expired : in std_logic;
        state : out std_logic_vector(1 downto 0);
        lights : out std_logic_vector(2 downto 0)  -- [red, yellow, green]
    );
end entity;

architecture behavioral of traffic_light_fsm is
    signal current_state : state_type;
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                current_state <= IDLE;
            else
                case current_state is
                    when IDLE =>
                        if timer_expired = '1' then
                            current_state <= GREEN;
                        end if;
                    when GREEN =>
                        if timer_expired = '1' then
                            current_state <= YELLOW;
                        end if;
                    when YELLOW =>
                        if timer_expired = '1' then
                            current_state <= RED;
                        end if;
                    when RED =>
                        if timer_expired = '1' then
                            current_state <= GREEN;
                        end if;
                end case;
            end if;
        end if;
    end process;
    
    -- Output logic
    process(current_state)
    begin
        case current_state is
            when IDLE | RED => lights <= "100";  -- Red only
            when GREEN => lights <= "001";       -- Green only
            when YELLOW => lights <= "010";      -- Yellow only
        end case;
    end process;
    
    -- State encoding
    with current_state select
        state <= "00" when IDLE,
                 "01" when GREEN,
                 "10" when YELLOW,
                 "11" when RED;
end architecture;
""",
    properties=[
        "reset -> next(state) == IDLE",
        "state == IDLE && timer_expired -> next(state) == GREEN",
        "state == GREEN && timer_expired -> next(state) == YELLOW",
        "state == YELLOW && timer_expired -> next(state) == RED",
        "state == RED && timer_expired -> next(state) == GREEN",
        "state == GREEN -> lights == 3'b001",
        "state == YELLOW -> lights == 3'b010",
        "state == RED || state == IDLE -> lights == 3'b100",
        "count_ones(lights) == 1",  # Exactly one light on
        "always(eventually(state == GREEN))"  # Liveness property
    ],
    should_verify=True,
    complexity="complex",
    circuit_type="sequential",
    description="Traffic light finite state machine with timer control"
)

# Circuits that should fail verification (for negative testing)
BUGGY_ADDER = CircuitFixture(
    name="buggy_adder",
    verilog_code="""
module buggy_adder(
    input [3:0] a,
    input [3:0] b,
    output [4:0] sum
);
    // Bug: missing carry bit
    assign sum = a + b + 1;  // Off by one error
endmodule
""",
    vhdl_code="""
entity buggy_adder is
    port (
        a : in std_logic_vector(3 downto 0);
        b : in std_logic_vector(3 downto 0);
        sum : out std_logic_vector(4 downto 0)
    );
end entity;

architecture behavioral of buggy_adder is
begin
    -- Bug: off by one error
    sum <= std_logic_vector(unsigned('0' & a) + unsigned('0' & b) + 1);
end architecture;
""",
    properties=[
        "sum == a + b",  # This should fail
        "sum >= a",
        "sum >= b"
    ],
    should_verify=False,  # Expected to fail
    complexity="simple",
    circuit_type="combinational",
    description="Adder with off-by-one bug for negative testing"
)

RACE_CONDITION_COUNTER = CircuitFixture(
    name="race_condition_counter",
    verilog_code="""
module race_condition_counter(
    input clk,
    input reset,
    output reg [7:0] count
);
    // Bug: Race condition in reset logic
    always @(posedge clk or posedge reset) begin
        if (reset)
            count <= 8'b0;
        else
            count <= count + 1;
    end
    
    // Bug: Another always block modifying same signal
    always @(negedge clk) begin
        if (!reset)
            count <= count + 1;  // Race condition!
    end
endmodule
""",
    vhdl_code="""
entity race_condition_counter is
    port (
        clk : in std_logic;
        reset : in std_logic;
        count : out std_logic_vector(7 downto 0)
    );
end entity;

architecture buggy of race_condition_counter is
    signal count_reg : unsigned(7 downto 0);
begin
    -- Multiple processes driving same signal (race condition)
    process(clk, reset)
    begin
        if reset = '1' then
            count_reg <= (others => '0');
        elsif rising_edge(clk) then
            count_reg <= count_reg + 1;
        end if;
    end process;
    
    process(clk)  -- Bug: Another process modifying count_reg
    begin
        if falling_edge(clk) and reset = '0' then
            count_reg <= count_reg + 1;  -- Race condition!
        end if;
    end process;
    
    count <= std_logic_vector(count_reg);
end architecture;
""",
    properties=[
        "reset -> next(count) == 0",
        "!reset -> next(count) == count + 1"  # This should fail due to race
    ],
    should_verify=False,  # Expected to fail
    complexity="medium",
    circuit_type="sequential",
    description="Counter with race condition for negative testing"
)

# Collection of all fixtures
ALL_FIXTURES: List[CircuitFixture] = [
    SIMPLE_ADDER,
    MULTIPLEXER_2TO1,
    DECODER_2TO4,
    COUNTER_8BIT,
    SHIFT_REGISTER,
    FSM_TRAFFIC_LIGHT,
    BUGGY_ADDER,
    RACE_CONDITION_COUNTER
]

# Categorized fixtures
SIMPLE_FIXTURES = [f for f in ALL_FIXTURES if f.complexity == "simple"]
MEDIUM_FIXTURES = [f for f in ALL_FIXTURES if f.complexity == "medium"]
COMPLEX_FIXTURES = [f for f in ALL_FIXTURES if f.complexity == "complex"]

COMBINATIONAL_FIXTURES = [f for f in ALL_FIXTURES if f.circuit_type == "combinational"]
SEQUENTIAL_FIXTURES = [f for f in ALL_FIXTURES if f.circuit_type == "sequential"]

VALID_FIXTURES = [f for f in ALL_FIXTURES if f.should_verify]
BUGGY_FIXTURES = [f for f in ALL_FIXTURES if not f.should_verify]

# Fixture lookup by name
FIXTURES_BY_NAME: Dict[str, CircuitFixture] = {f.name: f for f in ALL_FIXTURES}