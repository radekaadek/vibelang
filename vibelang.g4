grammar vibelang;

program: (statement | functionDef | structDef)* EOF;

structDef
    : 'struct' ID structField* 'end' # StructDefinition
    ;

structField
    : type ID ';'
    ;

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
    | lvalue '=' expr ';'                       # VarAssign
    | 'print' '(' expr ')' ';'              # PrintStmt
    | 'read' '(' ID ')' ';'                 # ReadStmt
    | 'if' expr 'then' statement* ('else' statement*)? 'end' # IfStmt
    | 'while' expr 'do' statement* 'end'    # WhileStmt
    | 'return' expr? ';'                    # ReturnStmt
    | ID '(' arguments? ')' ';'             # CallStmt
    ;

lvalue
    : ID ('.' ID)*
    ;

arguments
    : expr (',' expr)*
    ;

type: 'int32' | 'int64' | 'float32' | 'float64' | 'bool' | ID;

expr
    // Multiplication and division have higher precedence
    : expr '.' ID                               # MemberAccessExpr 
    | 'not' expr                                # NotExpr
    | expr ('*' | '/') expr                     # MulDivExpr
    | expr ('+' | '-') expr                     # AddSubExpr
    | expr ('<' | '<=' | '>' | '>=' | '==' | '!=') expr # RelExpr
    | expr 'and' expr                           # AndExpr
    | expr 'xor' expr                           # XorExpr
    | expr 'or' expr                            # OrExpr
    | ID '{' fieldInitList? '}'                 # StructInitExpr  
    | ID '(' arguments? ')'                     # CallExpr
    | ID                                        # IdExpr
    | INT                                       # IntExpr
    | FLOAT                                     # FloatExpr
    | BOOL                                      # BoolExpr
    | '(' expr ')'                              # ParenExpr
    ;

fieldInitList
    : ID ':' expr (',' ID ':' expr)*
    ;

// Lexer rules
BOOL: 'true' | 'false';
FLOAT: [0-9]+ '.' [0-9]+;
INT: [0-9]+;
ID: [a-zA-Z_][a-zA-Z0-9_]*;

LINE_COMMENT : '//' ~[\r\n]* -> skip;
BLOCK_COMMENT: '/*' .* '*/' -> skip;

WS: [ \t\r\n]+ -> skip;
