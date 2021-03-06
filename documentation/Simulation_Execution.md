Simulation Execution
==
System Simulations are executed with the execution engine executor (`cadCAD.engine.Executor`) given System Model 
Configurations. There are multiple simulation Execution Modes and Execution Contexts.

### Steps:
1. #### *Choose Execution Mode*:
    * ##### Simulation Execution Modes:
        `cadCAD` executes a process per System Model Configuration and a thread per System Simulation.
    ##### Class: `cadCAD.engine.ExecutionMode`
    ##### Attributes:
    * **Single Process:** A single process Execution Mode for a single System Model Configuration (Example: 
    `cadCAD.engine.ExecutionMode().single_proc`).
    * **Multi-Process:** Multiple process Execution Mode for System Model Simulations which executes on a thread per 
    given System Model Configuration (Example: `cadCAD.engine.ExecutionMode().multi_proc`).
2. #### *Create Execution Context using Execution Mode:*
```python
from cadCAD.engine import ExecutionMode, ExecutionContext
exec_mode = ExecutionMode()
single_proc_ctx = ExecutionContext(context=exec_mode.single_proc)
```
3. #### *Create Simulation Executor*
```python
from cadCAD.engine import Executor
from cadCAD import configs
simulation = Executor(exec_context=single_proc_ctx, configs=configs)
```
4. #### *Execute Simulation: Produce System Event Dataset*
A Simulation execution produces a System Event Dataset and the Tensor Field applied to initial states used to create it. 
```python
import pandas as pd
raw_system_events, tensor_field = simulation.execute()

# Simulation Result Types:
# raw_system_events: List[dict] 
# tensor_field: pd.DataFrame

# Result System Events DataFrame
simulation_result = pd.DataFrame(raw_system_events)
```

##### Example Tensor Field
```
+----+-----+--------------------------------+--------------------------------+
|    |   m | b1                             | s1                             |
|----+-----+--------------------------------+--------------------------------|
|  0 |   1 | <function p1m1 at 0x10c458ea0> | <function s1m1 at 0x10c464510> |
|  1 |   2 | <function p1m2 at 0x10c464048> | <function s1m2 at 0x10c464620> |
|  2 |   3 | <function p1m3 at 0x10c464400> | <function s1m3 at 0x10c464730> |
+----+-----+--------------------------------+--------------------------------+
```

##### Example Result: System Events DataFrame
```
+----+-------+------------+-----------+------+-----------+
|    |   run |   timestep |   substep |   s1 | s2        |
|----+-------+------------+-----------+------+-----------|
|  0 |     1 |          0 |         0 |    0 | 0.0       |
|  1 |     1 |          1 |         1 |    1 | 4         |
|  2 |     1 |          1 |         2 |    2 | 6         |
|  3 |     1 |          1 |         3 |    3 | [ 30 300] |
|  4 |     2 |          0 |         0 |    0 | 0.0       |
|  5 |     2 |          1 |         1 |    1 | 4         |
|  6 |     2 |          1 |         2 |    2 | 6         |
|  7 |     2 |          1 |         3 |    3 | [ 30 300] |
+----+-------+------------+-----------+------+-----------+
```

### Execution Examples:
##### Single Simulation Execution (Single Process Execution)
Example [System Model Configurations](link): 
* [System Model A](link): `/documentation/examples/sys_model_A.py`
* [System Model B](link): `/documentation/examples/sys_model_B.py`
Example Simulation Executions:
* [System Model A](link): `/documentation/examples/sys_model_A_exec.py`
* [System Model B](link): `/documentation/examples/sys_model_B_exec.py`
```python
import pandas as pd
from tabulate import tabulate
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from documentation.examples import sys_model_A
from cadCAD import configs

exec_mode = ExecutionMode()

# Single Process Execution using a Single System Model Configuration:
# sys_model_A
sys_model_A = [configs[0]] # sys_model_A
single_proc_ctx = ExecutionContext(context=exec_mode.single_proc)
sys_model_A_simulation = Executor(exec_context=single_proc_ctx, configs=sys_model_A)

sys_model_A_raw_result, sys_model_A_tensor_field = sys_model_A_simulation.execute()
sys_model_A_result = pd.DataFrame(sys_model_A_raw_result)
print()
print("Tensor Field: sys_model_A")
print(tabulate(sys_model_A_tensor_field, headers='keys', tablefmt='psql'))
print("Result: System Events DataFrame")
print(tabulate(sys_model_A_result, headers='keys', tablefmt='psql'))
print()
```

##### Multiple Simulation Execution

* ##### *Multi Process Execution*
Documentation: [Simulation Execution](link) 
[Example Simulation Executions::](link) `/documentation/examples/sys_model_AB_exec.py`
Example [System Model Configurations](link): 
* [System Model A](link): `/documentation/examples/sys_model_A.py`
* [System Model B](link): `/documentation/examples/sys_model_B.py`
```python
import pandas as pd
from tabulate import tabulate
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from documentation.examples import sys_model_A, sys_model_B
from cadCAD import configs

exec_mode = ExecutionMode()

# # Multiple Processes Execution using Multiple System Model Configurations:
# # sys_model_A & sys_model_B
multi_proc_ctx = ExecutionContext(context=exec_mode.multi_proc)
sys_model_AB_simulation = Executor(exec_context=multi_proc_ctx, configs=configs)

i = 0
config_names = ['sys_model_A', 'sys_model_B']
for sys_model_AB_raw_result, sys_model_AB_tensor_field in sys_model_AB_simulation.execute():
    sys_model_AB_result = pd.DataFrame(sys_model_AB_raw_result)
    print()
    print(f"Tensor Field: {config_names[i]}")
    print(tabulate(sys_model_AB_tensor_field, headers='keys', tablefmt='psql'))
    print("Result: System Events DataFrame:")
    print(tabulate(sys_model_AB_result, headers='keys', tablefmt='psql'))
    print()
    i += 1
```

* ##### *Parameter Sweep*
Documentation: [System Model Parameter Sweep](link) 
[Example:](link) `/documentation/examples/param_sweep.py`
```python
import pandas as pd
from tabulate import tabulate
# The following imports NEED to be in the exact order
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from documentation.examples import param_sweep
from cadCAD import configs

exec_mode = ExecutionMode()
multi_proc_ctx = ExecutionContext(context=exec_mode.multi_proc)
run = Executor(exec_context=multi_proc_ctx, configs=configs)

for raw_result, tensor_field in run.execute():
    result = pd.DataFrame(raw_result)
    print()
    print("Tensor Field:")
    print(tabulate(tensor_field, headers='keys', tablefmt='psql'))
    print("Output:")
    print(tabulate(result, headers='keys', tablefmt='psql'))
    print()
```
