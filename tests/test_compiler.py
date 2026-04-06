import subprocess
from pathlib import Path

import click
import pytest

from scripts.compiler import compile_vibe

# Ścieżka do katalogu z przykładami w Twoim projekcie
EXAMPLES_DIR = Path("examples")

# Słownik mapujący nazwy plików na oczekiwane standardowe wyjście po ich wykonaniu.
# Ponieważ używasz %f i %lf w formacie C dla float/double, domyślnie wyświetlają one 6 miejsc po przecinku.
EXPECTED_OUTPUTS = {
    "simple.vibe": "65\n45\n",
    "variables.vibe": "314.150000\n78.537500\n",
    "precedence.vibe": "20\n60.000000\n4.000000\n",
    "mixed_operators.vibe": "3\n3.333333\n",
    "casting.vibe": "3\n42.000000\n45.000000\n3\n3.000000\n500\n",
    "promotion.vibe": "110\n101.500000\n3.750000\n44.000000\n",
    "float.vibe": "78.537500\n",
    # Zaokrąglenia mogą się delikatnie różnić dla printf_formatting, ale poniższe wartości są typowe dla float64 w LLVM
    "printf_formatting.vibe": "42\n22\n3.140000\n2.718282\n",
}


@pytest.mark.parametrize("example_file, expected_output", EXPECTED_OUTPUTS.items())
def test_valid_examples(example_file: str, expected_output: str, tmp_path: Path) -> None:
    """
    Test weryfikuje poprawne działanie operacji arytmetycznych, przypisań i wejścia-wyjścia.
    """
    vibe_file = EXAMPLES_DIR / example_file
    output_ll = tmp_path / "output.ll"

    # 1. Kompilacja kodu .vibe do LLVM IR za pomocą funkcji compile_vibe
    compile_vibe(input_file=vibe_file, output=output_ll, opt_level=3, verbose=False)

    # 2. Wykonanie wygenerowanego kodu LLVM za pomocą interpretera lli
    run_process = subprocess.run(
        ["lli", str(output_ll)], capture_output=True, text=True
    )
    assert run_process.returncode == 0, (
        f"Błąd wykonania dla {example_file}:\n{run_process.stderr}"
    )

    # 3. Weryfikacja zgodności wyjścia (zmienne, operacje, rzutowania) z oczekiwanym wynikiem
    assert run_process.stdout == expected_output


def test_semantic_error_redeclaration(tmp_path):
    """
    Test weryfikujący czy kompilator poprawnie wychwytuje błędy semantyczne.
    Oparty na pliku redeclaration.vibe.
    """
    vibe_file = EXAMPLES_DIR / "redeclaration.vibe"
    output_ll = tmp_path / "output.ll"

    # Uruchomienie kompilatora dla kodu z celowym błędem
    # compile_vibe przechwytuje wyjątki i rzuca click.Abort
    with pytest.raises(click.Abort) as exc_info:
        compile_vibe(input_file=vibe_file, output=output_ll, opt_level=3, verbose=False)

    # Sprawdzenie czy w oryginalnym wyjątku (SemanticError) znalazła się odpowiednia wiadomość
    expected_error_msg = (
        "Semantic error: Variable 'licznik' already exists in this scope."
    )

    assert expected_error_msg in str(exc_info.value.__cause__)
