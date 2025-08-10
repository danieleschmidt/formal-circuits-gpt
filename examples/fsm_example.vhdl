-- Simple Finite State Machine in VHDL
library ieee;
use ieee.std_logic_1164.all;

entity fsm_example is
    port (
        clk : in std_logic;
        reset : in std_logic;
        start : in std_logic;
        done : out std_logic;
        state_out : out std_logic_vector(1 downto 0)
    );
end fsm_example;

architecture behavioral of fsm_example is
    type state_type is (IDLE, WORK, FINISH);
    signal current_state, next_state : state_type;
    
begin
    -- State register
    process(clk, reset)
    begin
        if reset = '1' then
            current_state <= IDLE;
        elsif rising_edge(clk) then
            current_state <= next_state;
        end if;
    end process;
    
    -- Next state logic
    process(current_state, start)
    begin
        case current_state is
            when IDLE =>
                if start = '1' then
                    next_state <= WORK;
                else
                    next_state <= IDLE;
                end if;
                
            when WORK =>
                next_state <= FINISH;
                
            when FINISH =>
                next_state <= IDLE;
                
            when others =>
                next_state <= IDLE;
        end case;
    end process;
    
    -- Output logic
    done <= '1' when current_state = FINISH else '0';
    
    -- State encoding for verification
    with current_state select
        state_out <= "00" when IDLE,
                    "01" when WORK, 
                    "10" when FINISH,
                    "11" when others;
                    
end behavioral;

-- Expected properties:
-- 1. Always starts in IDLE after reset
-- 2. Transitions IDLE -> WORK -> FINISH -> IDLE
-- 3. done is only high in FINISH state
-- 4. No deadlocks or unreachable states