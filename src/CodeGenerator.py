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
        self.i64 = ir.IntType(64)
        self.f32 = ir.FloatType()
        self.f64 = ir.DoubleType()

        self.type_map = {
            "int32": self.i32,
            "int64": self.i64,
            "float32": self.f32,
            "float64": self.f64,
        }

        # Printf function (external C function)
        printf_ty = ir.FunctionType(
            self.i32, [ir.IntType(8).as_pointer()], var_arg=True
        )
        self.printf = ir.Function(self.module, printf_ty, name="printf")

        self.fmt_int32 = self.create_fmt_string("fmt_int32", "%d")
        self.fmt_int64 = self.create_fmt_string("fmt_int64", "%lld")
        self.fmt_float32 = self.create_fmt_string("fmt_float32", "%f")
        self.fmt_float64 = self.create_fmt_string("fmt_float64", "%lf")

    def create_fmt_string(self, name: str, fmt: str) -> ir.GlobalVariable:
        fmt_str = f"{fmt}\n\0"
        c_fmt = ir.Constant(
            ir.ArrayType(ir.IntType(8), len(fmt_str)), bytearray(fmt_str.encode("utf8"))
        )
        glob = ir.GlobalVariable(self.module, c_fmt.type, name=name)
        glob.linkage = "internal"
        glob.global_constant = True
        glob.initializer = c_fmt
        return glob

    def cast_to(self, val: ir.Value, target_type: ir.Type) -> ir.Value:  # noqa: PLR0911
        """Automatic upcasting/downcasting to target type."""
        if self.builder is None:
            msg = "Semantic error: Cannot cast without a builder."
            raise SemanticError(msg)

        if val.type == target_type:
            return val

        # Int <-> Int
        if isinstance(val.type, ir.IntType) and isinstance(target_type, ir.IntType):
            if target_type.width > val.type.width:
                return self.builder.sext(val, target_type)  # pyright: ignore[reportReturnType]
            return self.builder.trunc(val, target_type)  # pyright: ignore[reportReturnType]

        # Float <-> Float
        if isinstance(val.type, (ir.FloatType, ir.DoubleType)) and isinstance(
            target_type, (ir.FloatType, ir.DoubleType)
        ):
            if target_type == self.f64:
                return self.builder.fpext(val, target_type)  # pyright: ignore[reportReturnType]
            return self.builder.fptrunc(val, target_type)  # pyright: ignore[reportReturnType]

        # Int -> Float
        if isinstance(val.type, ir.IntType) and isinstance(
            target_type, (ir.FloatType, ir.DoubleType)
        ):
            return self.builder.sitofp(val, target_type)  # pyright: ignore[reportReturnType]

        # Float -> Int
        if isinstance(val.type, (ir.FloatType, ir.DoubleType)) and isinstance(
            target_type, ir.IntType
        ):
            return self.builder.fptosi(val, target_type)  # pyright: ignore[reportReturnType]

        return val

    def promote_types(
        self, left: ir.Value, right: ir.Value
    ) -> tuple[ir.Value, ir.Value]:
        """Returns types of highest common precision."""
        if left.type == right.type:
            return left, right

        types = {left.type, right.type}
        # Precision hierarchy: float64 > float32 > int64 > int32
        if self.f64 in types:
            target = self.f64
        elif self.f32 in types:
            target = self.f32
        elif self.i64 in types:
            target = self.i64
        else:
            target = self.i32

        return self.cast_to(left, target), self.cast_to(right, target)

    # --- SCOPE MANAGEMENT (Symbol Table) ---
    def current_scope(self) -> dict[str, dict[str, object]]:
        """Return the current symbol table."""
        return self.symbol_table[-1]

    def allocate_variable(self, name: str, var_type: str, line: int) -> ir.AllocaInstr:
        """Allocate memory for a variable."""
        if name in self.current_scope():
            raise SemanticError(
                f"Semantic error: Variable '{name}' already exists in this scope.", line
            )

        if var_type not in self.type_map:
            raise SemanticError(f"Semantic error: Unsupported type '{var_type}'.", line)

        llvm_type = self.type_map[var_type]
        ptr = self.builder.alloca(llvm_type, name=name)  # pyright: ignore[reportOptionalMemberAccess]
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

        # Determine target type

        target_llvm_type = self.type_map[var_type]
        val = self.cast_to(val, target_llvm_type)  # pyright: ignore[reportArgumentType]

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

        target_llvm_type = self.type_map[str(var_info["type"])]
        val = self.cast_to(val, target_llvm_type)  # pyright: ignore[reportArgumentType]

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
            fmt_ptr = self.builder.bitcast(self.fmt_float64, ir.IntType(8).as_pointer())
        elif val.type == self.f32:
            fmt_ptr = self.builder.bitcast(self.fmt_float32, ir.IntType(8).as_pointer())
            val = self.builder.fpext(
                val, self.f64
            )  # C expects a float as a double in printf
        elif val.type == self.i64:
            fmt_ptr = self.builder.bitcast(self.fmt_int64, ir.IntType(8).as_pointer())
        else:  # i32
            fmt_ptr = self.builder.bitcast(self.fmt_int32, ir.IntType(8).as_pointer())

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

        self.type_map = {
            "int32": self.i32,
            "int64": self.i64,
            "float32": self.f32,
            "float64": self.f64,
        }
        typ = self.type_map[var_info["type"]]  # pyright: ignore[reportArgumentType]
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
        val = float(ctx.FLOAT().getText())  # pyright: ignore[reportOptionalMemberAccess]
        return ir.Constant(self.f64, val)

    @override
    def visitParenExpr(self, ctx: vibelangParser.ParenExprContext) -> object:
        return self.visit(ctx.expr())  # pyright: ignore[reportArgumentType]

    @override
    def visitAddSubExpr(self, ctx: vibelangParser.AddSubExprContext) -> object:
        if ctx.start is None:
            msg = "Semantic error: Cannot recognize line number."
            raise SemanticError(msg)

        if self.builder is None:
            msg = "Semantic error: Builder is not initialized."
            raise SemanticError(msg, ctx.start.line)
        left = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]
        op = ctx.getChild(1).getText()

        left, right = self.promote_types(left, right)  # pyright: ignore[reportArgumentType]
        is_float = isinstance(left.type, (ir.FloatType, ir.DoubleType))

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

        left = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]
        op = ctx.getChild(1).getText()

        left, right = self.promote_types(left, right)  # pyright: ignore[reportArgumentType]
        is_float = isinstance(left.type, (ir.FloatType, ir.DoubleType))

        if op == "*":
            if is_float:
                return self.builder.fmul(left, right, name="fmultmp")
            return self.builder.mul(left, right, name="multmp")
        if is_float:
            return self.builder.fdiv(left, right, name="fdivtmp")
        return self.builder.sdiv(left, right, name="divtmp")
