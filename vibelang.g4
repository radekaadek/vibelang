grammar vibelang;

program: (statement | functionDef)* EOF;

functionDef
    : 'func' ID '(' parameters? ')' '->' returnType statement* 'end' # FunctionDefinition
    ;

parameters
    : parameter (',' parameter)*
    ;

parameter
    : type ID
    ;

returnType
    : type | 'void'
    ;

statement
    : type ID '=' expr ';'                  # VarDeclAssign
    | ID '=' expr ';'                       # VarAssign
    | 'print' '(' expr ')' ';'              # PrintStmt
    | 'read' '(' ID ')' ';'                 # ReadStmt
    | 'if' expr 'then' statement* ('else' statement*)? 'end' # IfStmt
    | 'while' expr 'do' statement* 'end'    # WhileStmt
    | 'return' expr? ';'                    # ReturnStmt
    | ID '(' arguments? ')' ';'             # CallStmt
    ;

arguments
    : expr (',' expr)*
    ;

type: 'int32' | 'int64' | 'float32' | 'float64' | 'bool';

expr
    // Multiplication and division have higher precedence
    : 'not' expr                # NotExpr
    | expr ('*' | '/') expr     # MulDivExpr
    | expr ('+' | '-') expr     # AddSubExpr
    | expr ('<' | '<=' | '>' | '>=' | '==' | '!=') expr # RelExpr
    | expr 'and' expr           # AndExpr
    | expr 'xor' expr           # XorExpr
    | expr 'or' expr            # OrExpr
    | ID                        # IdExpr
    | INT                       # IntExpr
    | FLOAT                     # FloatExpr
    | BOOL                      # BoolExpr
    | '(' expr ')'              # ParenExpr
    | ID '(' arguments? ')' ';' # CallExpr
    ;

// Lexer rules
BOOL: 'true' | 'false';
FLOAT: [0-9]+ '.' [0-9]+;
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;
WS: [ \t\r\n]+ -> skip;
