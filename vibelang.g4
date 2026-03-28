grammar vibelang;

program: statement* EOF;

statement
    : type ID '=' expr ';'      # VarDeclAssign
    | ID '=' expr ';'           # VarAssign
    | 'print' '(' expr ')' ';'  # PrintStmt
    ;

type: 'int';

expr
    : expr ('+' | '-') expr     # AddSubExpr
    | ID                        # IdExpr
    | INT                       # IntExpr
    ;

// Lexer rules
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
WS: [ \t\r\n]+ -> skip;
