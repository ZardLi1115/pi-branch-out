Implement `chat.utils.run_cmd` in `/app/chat/utils.py`.

Requirement from EvoCodeBench `chat.utils.run_cmd`:

- Functionality: Executes a given bash command by printing it first and then using the system's command line interface to run it.
- Arguments: `cmd: str` is the bash command to execute.
- Return: `int` exit status. `0` typically means success; any other value indicates an error.

Keep the existing function signature. Do not add extra files unless needed. After implementing, make sure `python3 -c "from chat.utils import run_cmd; assert run_cmd('true') == 0"` works from `/app`.
