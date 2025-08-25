pytest -vv -s -o log_cli=true --log-cli-level=INFO tests/parallel/test_registry.py::test_load_model_with_env

ps aux | grep micromamba
pkill -9 micromamba
rm -f ~/.mamba/pkgs/locks/\*
