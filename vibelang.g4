grammar vibelang;

program: statement* EOF;

statement
    : type ID '=' expr ';'              # VarDeclAssign
    | ID '=' expr ';'                   # VarAssign
    | 'print' '(' expr ')' ';'          # PrintStmt
    | 'read' '(' ID ')' ';'             # ReadStmt
    | 'if' expr 'then' statement* ('else' statement*)? 'end' # IfStmt
    ;

type: 'int32' | 'int64' | 'float32' | 'float64' | 'bool';

expr
    // Multiplication and division have higher precedence
    : 'not' expr                # NotExpr
    | expr ('*' | '/') expr     # MulDivExpr
    | expr ('+' | '-') expr     # AddSubExpr
    | expr 'and' expr           # AndExpr
    | expr 'xor' expr           # XorExpr
    | expr 'or' expr            # OrExpr
    | ID                        # IdExpr
    | INT                       # IntExpr
    | FLOAT                     # FloatExpr
    | BOOL                      # BoolExpr
    | '(' expr ')'              # ParenExpr
    ;

// Lexer rules
BOOL: 'true' | 'false';
FLOAT: [0-9]+ '.' [0-9]+;
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
WS: [ \t\r\n]+ -> skip;
