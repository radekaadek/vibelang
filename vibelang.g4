grammar vibelang;

program: (statement | functionDef | structDef | classDef)* EOF;

classDef
    : 'class' ID classMember* 'end' # ClassDefinition
    ;

classMember
    : type ID ';'                                                           # ClassField
    | 'func' ID '(' parameters? ')' '->' returnType statement* 'end'        # ClassMethod
    ;

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
    | lvalue '=' expr ';'                   # VarAssign
    | 'print' '(' expr ')' ';'              # PrintStmt
    | 'read' '(' ID ')' ';'                 # ReadStmt
    | 'if' expr 'then' statement* ('else' statement*)? 'end' # IfStmt
    | 'while' expr 'do' statement* 'end'    # WhileStmt
    | 'return' expr? ';'                    # ReturnStmt
    | ID '(' arguments? ')' ';'             # CallStmt
    | expr '.' ID '(' arguments? ')' ';'    # MethodCallStmt
    ;

lvalue
    : ('self' | ID) ('.' ID)*
    ;

arguments
    : expr (',' expr)*
    ;

type: 'int32' | 'int64' | 'float32' | 'float64' | 'bool' | ID;

expr
    // Multiplication and division have higher precedence
    : expr '.' ID '(' arguments? ')'            # MethodCallExpr
    | expr '.' ID                               # MemberAccessExpr 
    | 'not' expr                                # NotExpr
    | expr ('*' | '/') expr                     # MulDivExpr
    | expr ('+' | '-') expr                     # AddSubExpr
    | expr ('<' | '<=' | '>' | '>=' | '==' | '!=') expr # RelExpr
    | expr 'and' expr                           # AndExpr
    | expr 'xor' expr                           # XorExpr
    | expr 'or' expr                            # OrExpr
    | 'new' ID '(' arguments? ')'               # NewObjectExpr
    | ID '{' fieldInitList? '}'                 # StructInitExpr  
    | ID '(' arguments? ')'                     # CallExpr
    | 'self'                                    # SelfExpr
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
BLOCK_COMMENT: '/*' .*? '*/' -> skip;

WS: [ \t\r\n]+ -> skip;
