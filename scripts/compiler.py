from pathlib import Path

import click
from antlr4 import CommonTokenStream, InputStream
from llvmlite import binding

from src.CodeGenerator import CodeGenerator
from vibelangLexer import vibelangLexer
from vibelangParser import vibelangParser

# Initialize LLVM native targets
binding.initialize_native_target()
binding.initialize_native_asmprinter()


def compile_vibe(
    input_file: Path, output: Path, opt_level: int, *, verbose: bool
) -> None:
    """Vibelang Compiler 🚀

    Compiles an INPUT_FILE written in Vibelang into optimized LLVM IR.
    """
    click.secho(f"🛠️  Compiling '{input_file.name}'...", fg="cyan", bold=True)

    try:
        # 0. Read source code
        code = input_file.read_text(encoding="utf-8")

        # 1. Lexing and Parsing
        click.secho("parsing...", fg="yellow", dim=True)
        input_stream = InputStream(code)
        lexer = vibelangLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = vibelangParser(stream)
        tree = parser.program()

        # 2. Generating IR
        click.secho("generating ir...", fg="yellow", dim=True)
        compiler = CodeGenerator()
        compiler.visit(tree)
        unoptimized_ir = str(compiler.module)

        if verbose:
            click.secho("\n----- Generated LLVM IR (Unoptimized) -----", fg="magenta")
            click.echo(unoptimized_ir)

        # 3. LLVM Setup and Verification
        mod = binding.parse_assembly(unoptimized_ir)
        mod.verify()

        # 4. Target Machine Setup
        target_triple = binding.get_process_triple()
        target = binding.Target.from_triple(target_triple)
        target_machine = target.create_target_machine()

        # 5. Optimizations
        click.secho(f"running passes (O{opt_level})...", fg="yellow", dim=True)
        pto = binding.create_pipeline_tuning_options(
            speed_level=opt_level, size_level=0
        )
        pass_builder = binding.create_pass_builder(target_machine, pto)
        mpm = pass_builder.getModulePassManager()

        mpm.run(mod, pass_builder)
        optimized_ir = str(mod)

        if verbose:
            click.secho("\n----- Optimized LLVM IR -----", fg="magenta")
            click.echo(optimized_ir)

        # 6. Save output
        with output.open("w") as f:
            f.write(optimized_ir)

        click.secho(f"✨ Success! Compiled to '{output}'.", fg="green", bold=True)

    except Exception as e:
        click.secho(f"\n❌ Compilation failed: {e}", fg="red", bold=True, err=True)
        raise click.Abort from e


@click.command()
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=Path("output.ll"),
    show_default=True,
    help="Path where the compiled LLVM IR will be saved.",
)
@click.option(
    "-O",
    "--opt-level",
    type=click.IntRange(0, 3),
    default=3,
    show_default=True,
    help="Optimization level (0 = none, 3 = max).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print the generated LLVM IR (both unoptimized and optimized) to the console.",
)
def compile_vibe_app(
    input_file: Path, output: Path, opt_level: int, *, verbose: bool
) -> None:
    compile_vibe(input_file, output, opt_level, verbose=verbose)


if __name__ == "__main__":
   compile_vibe_app()
