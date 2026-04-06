# vibelang compiler

## Installation

The compiler requires the user to have Python 3.14+, LLVM 22+ and Java 17+ installed on their system.

```bash
# Install ANTLR
curl -O https://www.antlr.org/download/antlr-4.13.1-complete.jar
alias antlr4='java -jar antlr-4.13.2-complete.jar'

# Generate Python parser code using the Visitor pattern
antlr4 -Dlanguage=Python3 -visitor -no-listener vibelang.g4

# Install Python requirements
pip install -r requirements.txt
```

## Compiling

```bash
python3 -m scripts.compiler -O 3 -v examples/simple.vibe
```

## Running

```bash
lli output.ll
```

or

```bash
clang output.ll -o vibe_program
chmod +x vibe_program
./vibe_program
```
