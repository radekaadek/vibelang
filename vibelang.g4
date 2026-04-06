grammar vibelang;

program: statement* EOF;

statement
    : type ID '=' expr ';'      # VarDeclAssign
    | ID '=' expr ';'           # VarAssign
    | 'print' '(' expr ')' ';'  # PrintStmt
    ;

type: 'int32' | 'int64' | 'float32' | 'float64';

expr
    // Multiplication and division have higher precedence
    : expr ('*' | '/') expr     # MulDivExpr
    | expr ('+' | '-') expr     # AddSubExpr
    | ID                        # IdExpr
    | INT                       # IntExpr
    | FLOAT                     # FloatExpr
    | '(' expr ')'              # ParenExpr
    ;

// Lexer rules
FLOAT: [0-9]+ '.' [0-9]+;
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
WS: [ \t\r\n]+ -> skip;
