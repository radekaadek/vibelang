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
    "logic.vibe": "0\n1\n0\n1\n0\n1\n0\n1\n0\n0\n",
    "if.vibe": "1\n1\n",
    "if_else.vibe": "1\n1\n0\n0\n",
    "relations.vibe": "1\n0\n1\n0\n1\n0\n1\n1\n0\n1\n0\n1\n1\n0\n1\n0\n1\n100\n200\n10\n5\n",
    "relations_type.vibe": "1\n1\n0\n1\n1\n1\n0\n1\n0\n1\n0\n1\n1\n",
    "relations_mixed.vibe": "1\n1\n0\n1\n0\n1\n0\n1\n1\n0\n999\n",
    "local_variables.vibe": "32\n5\n64\n64\n12\n24\n12\n24\n7\n",
    "function.vibe": "99999\n12\n99999\n120\n",
    "struct.vibe": "2500.000000\n10.500000\n20.000000\n100.000000\n1\n",
    "class.vibe": "1\n0.880000\n1000.000000\n120.500000\n45.000000\n0.990000\n100.000000\n",
    "program.vibe": "3628800\n610\n12\n1\n25\n150.000000\n1\n120.000000\n0\n2\n3\n5\n7\n11\n13\n17\n19\n23\n29\n10\n",
}


@pytest.mark.parametrize("example_file, expected_output", EXPECTED_OUTPUTS.items())
def test_valid_examples(
    example_file: str, expected_output: str, tmp_path: Path
) -> None:
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
    assert (
        run_process.returncode == 0
    ), f"Błąd wykonania dla {example_file}:\n{run_process.stderr}"

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

    # Zaktualizowana oczekiwana wiadomość błędu
    # expected_error_msg = "Semantic error: Variable 'licznik' already exists."
    expected_error_msg = (
        "Semantic error: Variable 'licznik' already exists in this scope."
    )

    assert expected_error_msg in str(exc_info.value.__cause__)


def test_read_statement(tmp_path: Path) -> None:
    """
    Test weryfikujący poprawne działanie instrukcji read() z klawiatury (stdin).
    Oparty na pliku examples/read.vibe.
    """
    # 1. Wskazanie na plik w katalogu examples
    vibe_file = EXAMPLES_DIR / "read.vibe"
    output_ll = tmp_path / "output.ll"

    # 2. Kompilacja do LLVM IR
    compile_vibe(input_file=vibe_file, output=output_ll, opt_level=3, verbose=False)

    # 3. Wykonanie programu z symulacją wejścia użytkownika
    # Symulujemy wpisanie 4 różnych wartości oddzielonych enterami (\n)
    simulated_user_input = "42\n8589934592\n3.14\n2.718281828\n"

    run_process = subprocess.run(
        ["lli", str(output_ll)],
        input=simulated_user_input,  # Przekazanie danych do strumienia wejściowego
        capture_output=True,
        text=True,
    )

    assert (
        run_process.returncode == 0
    ), f"Błąd wykonania instrukcji read():\n{run_process.stderr}"

    # 4. Weryfikacja zgodności wyjścia (floaty domyślnie wyświetlają 6 miejsc po przecinku)
    expected_output = "42\n8589934592\n3.140000\n2.718282\n"

    assert run_process.stdout == expected_output
