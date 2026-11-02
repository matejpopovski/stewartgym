# stewartgym

A six-actuator Stewart platform balancing a rolling ball in MuJoCo. This commit holds the
hand-written closed-chain model, its numerical validation against analytic kinematics, the task
environment and the cascaded PID baseline. Run the tests with `uv sync --frozen` then
`uv run --frozen pytest`.
