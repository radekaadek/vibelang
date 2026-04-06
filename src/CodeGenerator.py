from typing import cast, override

from antlr4 import TerminalNode
from llvmlite import binding, ir

from src.SemanticError import SemanticError
from vibelangParser import vibelangParser
from vibelangVisitor import vibelangVisitor

binding.initialize_native_target()
binding.initialize_native_asmprinter()


class CodeGenerator(vibelangVisitor):
    module: ir.Module
    builder: ir.IRBuilder | None
    symbol_table: list[dict[str, dict[str, object]]]
    i32: ir.IntType
    printf: ir.Function
    global_fmt_int: ir.GlobalVariable

    def __init__(self) -> None:
        """Initialize the LLVM module and target machine."""
        super().__init__()
        self.module = ir.Module(name="vibelangModule")

        target_triple = binding.get_process_triple()
        self.module.triple = target_triple

        target = binding.Target.from_triple(target_triple)
        target_machine = target.create_target_machine()
        self.module.data_layout = str(target_machine.target_data)

        self.builder = None

        self.symbol_table = [{}]

        # LLVM Types
        self.i32 = ir.IntType(32)
        self.f64 = ir.DoubleType()

        # Printf function (external C function)
        printf_ty = ir.FunctionType(
            self.i32, [ir.IntType(8).as_pointer()], var_arg=True
        )
        self.printf = ir.Function(self.module, printf_ty, name="printf")

        # Format string for printf ("%d\n" for int)
        fmt_str_int = "%d\n\0"
        c_fmt_int = ir.Constant(
            ir.ArrayType(ir.IntType(8), len(fmt_str_int)),
            bytearray(fmt_str_int.encode("utf8")),
        )
        self.global_fmt_int = ir.GlobalVariable(
            self.module, c_fmt_int.type, name="fmt_int"
        )
        self.global_fmt_int.linkage = "internal"
        self.global_fmt_int.global_constant = True
        self.global_fmt_int.initializer = c_fmt_int

        # Format string for printf ("%f\n" for float)
        fmt_str_float = "%f\n\0"
        c_fmt_float = ir.Constant(
            ir.ArrayType(ir.IntType(8), len(fmt_str_float)),
            bytearray(fmt_str_float.encode("utf8")),
        )
        self.global_fmt_float = ir.GlobalVariable(
            self.module, c_fmt_float.type, name="fmt_float"
        )
        self.global_fmt_float.linkage = "internal"
        self.global_fmt_float.global_constant = True
        self.global_fmt_float.initializer = c_fmt_float

    # --- SCOPE MANAGEMENT (Symbol Table) ---
    def current_scope(self) -> dict[str, dict[str, object]]:
        """Return the current symbol table."""
        return self.symbol_table[-1]

    def allocate_variable(self, name: str, var_type: str, line: int) -> ir.AllocaInstr:
        """Allocate memory for a variable."""
        if name in self.current_scope():
            msg = f"Semantic error: Variable '{name}' already exists in this scope."
            raise SemanticError(msg, line)

        if var_type == "int":
            llvm_type = self.i32
        elif var_type == "float":
            llvm_type = self.f64
        else:
            msg = f"Semantic error: Unsupported type '{var_type}'."
            raise SemanticError(msg, line)

        if self.builder is None:
            msg = "Semantic error: Cannot allocate memory."
            raise SemanticError(msg, line)

        ptr = self.builder.alloca(llvm_type, name=name)
        self.current_scope()[name] = {"ptr": ptr, "type": var_type}
        return ptr

    def lookup_variable(self, name: str, line: int) -> dict[str, object]:
        """Lookup a variable in the current scope."""
        for scope in reversed(self.symbol_table):
            if name in scope:
                return scope[name]
        msg = f"Semantic error: Undeclared variable '{name}'."
        raise SemanticError(msg, line)

    # --- AST NODE VISITING ---

    @override
    def visitProgram(self, ctx: vibelangParser.ProgramContext) -> object:
        # Main program function (main)
        func_type = ir.FunctionType(self.i32, [])
        func = ir.Function(self.module, func_type, name="main")
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Visit all instructions inside 'main'
        for child in ctx.getChildren():
            if not isinstance(child, TerminalNode):
                self.visit(child)

        # Return 0 at the end of main
        _ = self.builder.ret(ir.Constant(self.i32, 0))
        return None

    @override
    def visitVarDeclAssign(self, ctx: vibelangParser.VarDeclAssignContext) -> object:
        type_ctx = ctx.type_()
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        if type_ctx is None:
            msg = "Semantic error: Cannot recognize variable type."
            raise SemanticError(msg, ctx.start.line)
        var_type = type_ctx.getText()

        id_node = ctx.ID()
        if id_node is None:
            msg = "Semantic error: Cannot recognize variable name."
            raise SemanticError(msg, ctx.start.line)
        var_name = id_node.getText()

        ptr = self.allocate_variable(var_name, var_type, ctx.start.line)

        expr_ctx = ctx.expr()
        if expr_ctx is None:
            msg = "Semantic error: Cannot recognize expression."
            raise SemanticError(msg, ctx.start.line)
        val = self.visit(expr_ctx)

        if self.builder is None:
            msg = "Semantic error: Cannot allocate memory."
            raise SemanticError(msg, ctx.start.line)

        # Variable promotion
        if var_type == "float" and val.type == self.i32:
            val = self.builder.sitofp(val, self.f64)
        elif var_type == "int" and val.type == self.f64:
            val = self.builder.fptosi(val, self.i32)

        _ = self.builder.store(val, ptr)
        return None

    @override
    def visitVarAssign(self, ctx: vibelangParser.VarAssignContext) -> object:
        id_node = ctx.ID()
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        if id_node is None:
            msg = "Semantic error: Cannot recognize variable name."
            raise SemanticError(msg, ctx.start.line)
        var_name = id_node.getText()

        var_info = self.lookup_variable(var_name, ctx.start.line)
        ptr = cast("ir.Value", var_info["ptr"])

        expr_ctx = ctx.expr()
        if expr_ctx is None:
            msg = "Semantic error: Cannot recognize expression."
            raise SemanticError(msg, ctx.start.line)
        val = self.visit(expr_ctx)

        if self.builder is None:
            msg = "Semantic error: Cannot allocate memory."
            raise SemanticError(msg, ctx.start.line)

        if var_info["type"] == "float" and val.type == self.i32:
            val = self.builder.sitofp(val, self.f64)
        elif var_info["type"] == "int" and val.type == self.f64:
            val = self.builder.fptosi(val, self.i32)

        _ = self.builder.store(val, ptr)

        return None

    @override
    def visitPrintStmt(self, ctx: vibelangParser.PrintStmtContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        expr_ctx = ctx.expr()
        if expr_ctx is None:
            msg = "Semantic error: Cannot recognize expression."
            raise SemanticError(msg, ctx.start.line)
        val = self.visit(expr_ctx)

        if self.builder is None:
            msg = "Semantic error: Cannot allocate memory."
            raise SemanticError(msg, ctx.start.line)

        if val.type == self.f64:
            fmt_ptr = self.builder.bitcast(
                self.global_fmt_float, ir.IntType(8).as_pointer()
            )
            val = self.builder.fpext(val, ir.DoubleType())
        else:
            fmt_ptr = self.builder.bitcast(
                self.global_fmt_int, ir.IntType(8).as_pointer()
            )
            if val.type == ir.FloatType():
                val = self.builder.fpext(val, ir.DoubleType())

        _ = self.builder.call(self.printf, [fmt_ptr, val])
        return None

    @override
    def visitIdExpr(self, ctx: vibelangParser.IdExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        id_node = ctx.ID()
        if id_node is None:
            msg = "Semantic error: Cannot recognize variable name."
            raise SemanticError(msg, ctx.start.line)
        var_name = id_node.getText()

        var_info = self.lookup_variable(var_name, ctx.start.line)
        ptr = cast("ir.Value", var_info["ptr"])

        if self.builder is None:
            msg = "Semantic error: Cannot allocate memory."
            raise SemanticError(msg, ctx.start.line)

        typ = self.f64 if var_info["type"] == "float" else self.i32

        return self.builder.load(ptr, typ=typ, name=f"{var_name}_val")

    @override
    def visitIntExpr(self, ctx: vibelangParser.IntExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        int_node = ctx.INT()
        if int_node is None:
            msg = "Semantic error: Cannot recognize integer."
            raise SemanticError(msg, ctx.start.line)
        val = int(int_node.getText())
        return ir.Constant(self.i32, val)

    @override
    def visitFloatExpr(self, ctx: vibelangParser.FloatExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        if ctx.FLOAT() is None:
            msg = "Semantic error: Cannot recognize float."
            raise SemanticError(msg, ctx.start.line)
        if ctx.FLOAT() is None:
            msg = "Semantic error: Cannot recognize float."
            raise SemanticError(msg, ctx.start.line)
        val = float(ctx.FLOAT().getText()) # pyright: ignore[reportOptionalMemberAccess]
        return ir.Constant(self.f64, val)

    @override
    def visitParenExpr(self, ctx: vibelangParser.ParenExprContext) -> object:
        return self.visit(ctx.expr()) # pyright: ignore[reportArgumentType]

    @override
    def visitAddSubExpr(self, ctx: vibelangParser.AddSubExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)

        if self.builder is None:
            msg = "Semantic error: Builder is not initialized."
            raise SemanticError(msg, ctx.start.line)
        left = self.visit(ctx.expr(0)) # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1)) # pyright: ignore[reportArgumentType]
        op = ctx.getChild(1).getText()

        # Type promotion, if one of the arguments is float (int becomes float)
        is_float = self.f64 in (left.type, right.type)
        if left.type == self.i32 and right.type == self.f64:
            left = self.builder.sitofp(left, self.f64)
        if right.type == self.i32 and left.type == self.f64:
            right = self.builder.sitofp(right, self.f64)

        if op == "+":
            if is_float:
                return self.builder.fadd(left, right, name="faddtmp")
            return self.builder.add(left, right, name="addtmp")
        if is_float:
            return self.builder.fsub(left, right, name="fsubtmp")
        return self.builder.sub(left, right, name="subtmp")

    @override
    def visitMulDivExpr(self, ctx: vibelangParser.MulDivExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)
        if self.builder is None:
            msg = "Semantic error: Builder is not initialized."
            raise SemanticError(msg, ctx.start.line)

        left = self.visit(ctx.expr(0)) # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1)) # pyright: ignore[reportArgumentType]
        op = ctx.getChild(1).getText()

        is_float = self.f64 in (left.type, right.type)
        if left.type == self.i32 and right.type == self.f64:
            left = self.builder.sitofp(left, self.f64)
        if right.type == self.i32 and left.type == self.f64:
            right = self.builder.sitofp(right, self.f64)

        if op == "*":
            if is_float:
                return self.builder.fmul(left, right, name="fmultmp")
            return self.builder.mul(left, right, name="multmp")
        if is_float:
            return self.builder.fdiv(left, right, name="fdivtmp")
        return self.builder.sdiv(left, right, name="divtmp")
