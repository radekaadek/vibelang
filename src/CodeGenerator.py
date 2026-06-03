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
    symbol_table_stack: list[dict[str, dict[str, object]]]
    i32: ir.IntType
    printf: ir.Function

    def __init__(self) -> None:
        """Initialize the LLVM module and target machine."""
        super().__init__()

        llvm_context = ir.Context()
        self.module = ir.Module(name="vibelangModule", context=llvm_context)

        target_triple = binding.get_process_triple()
        self.module.triple = target_triple

        target = binding.Target.from_triple(target_triple)
        target_machine = target.create_target_machine()
        self.module.data_layout = str(target_machine.target_data)

        self.builder = None

        self.symbol_table_stack = [{}]

        # LLVM Types
        self.i1 = ir.IntType(1)
        self.i32 = ir.IntType(32)
        self.i64 = ir.IntType(64)
        self.f32 = ir.FloatType()
        self.f64 = ir.DoubleType()

        self.type_map = {
            "bool": self.i1,
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

        self.fmt_bool = self.create_fmt_string("fmt_bool", "%d")
        self.sfmt_bool = self.create_scanf_fmt_string("sfmt_bool", "%d")
        self.fmt_int32 = self.create_fmt_string("fmt_int32", "%d")
        self.fmt_int64 = self.create_fmt_string("fmt_int64", "%lld")
        self.fmt_float32 = self.create_fmt_string("fmt_float32", "%f")
        self.fmt_float64 = self.create_fmt_string("fmt_float64", "%lf")

        # Scanf function (external C function)
        scanf_ty = ir.FunctionType(self.i32, [ir.IntType(8).as_pointer()], var_arg=True)
        self.scanf = ir.Function(self.module, scanf_ty, name="scanf")

        self.sfmt_int32 = self.create_scanf_fmt_string("sfmt_int32", "%d")
        self.sfmt_int64 = self.create_scanf_fmt_string("sfmt_int64", "%lld")
        self.sfmt_float32 = self.create_scanf_fmt_string("sfmt_float32", "%f")
        self.sfmt_float64 = self.create_scanf_fmt_string("sfmt_float64", "%lf")

        # The return type of the function currently being emitted
        self.current_function_return_type: ir.Type | None = None

        #   {
        #     "type":   <LLVM identified struct type>,
        #     "fields": { field_name: {"index": <int>, "type": <type name str>} }
        #   }
        # The per-field "index" is the field's position in the struct, reused for
        # field address and field read.
        self.struct_info: dict[str, dict[str, object]] = {}

        # Same shape as struct_info, but ONLY classes are registered here.
        self.class_info: dict[str, dict[str, object]] = {}

    @property
    def current_scope(self) -> dict[str, dict[str, object]]:
        return self.symbol_table_stack[-1]

    def enter_scope(self) -> None:
        self.symbol_table_stack.append({})

    def exit_scope(self) -> None:
        _ = self.symbol_table_stack.pop()

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

    def create_scanf_fmt_string(self, name: str, fmt: str) -> ir.GlobalVariable:
        fmt_str = f"{fmt}\0"
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

    def allocate_variable(self, name: str, var_type: str, line: int) -> ir.AllocaInstr:
        """Allocate memory for a variable."""
        if name in self.current_scope:
            raise SemanticError(
                f"Semantic error: Variable '{name}' already exists in this scope.", line
            )

        if var_type not in self.type_map:
            raise SemanticError(f"Semantic error: Unsupported type '{var_type}'.", line)

        llvm_type = self.type_map[var_type]
        ptr = self.builder.alloca(llvm_type, name=name)  # pyright: ignore[reportOptionalMemberAccess]
        self.current_scope[name] = {"ptr": ptr, "type": var_type}
        return ptr

    def lookup_variable(self, name: str, line: int) -> dict[str, object]:
        """Lookup a variable."""
        for scope in reversed(self.symbol_table_stack):
            if name in scope:
                return scope[name]

        raise SemanticError(f"Semantic error: Undeclared variable '{name}'.", line)

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

        target_llvm_type = self.type_map[var_type]
        val = self.cast_to(val, target_llvm_type)  # pyright: ignore[reportArgumentType]

        _ = self.builder.store(val, ptr)
        return None

    @override
    def visitVarAssign(self, ctx: vibelangParser.VarAssignContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        lvalue_ctx = ctx.lvalue()
        # Parse 'self' or ID correctly depending on the first token
        first_token = lvalue_ctx.getChild(0).getText()  # pyright: ignore[reportOptionalMemberAccess]
        var_name = first_token

        var_info = self.lookup_variable(var_name, ctx.start.line)
        ptr = cast("ir.Value", var_info["ptr"])
        current_type_str = str(var_info["type"])

        id_nodes = lvalue_ctx.ID()  # pyright: ignore[reportOptionalMemberAccess]
        fields = (
            [n.getText() for n in id_nodes]
            if first_token == "self"
            else [n.getText() for n in id_nodes[1:]]
        )  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues, reportIndexIssue, reportOptionalSubscript]

        for field_name in fields:
            if current_type_str not in self.struct_info:
                raise SemanticError(
                    f"Semantic error: Type '{current_type_str}' is not a struct/class.",
                    ctx.start.line,
                )

            struct_def = cast("dict", self.struct_info[current_type_str])  # pyright: ignore[reportMissingTypeArgument]
            struct_fields = cast("dict", struct_def["fields"])  # pyright: ignore[reportMissingTypeArgument]

            if field_name not in struct_fields:
                raise SemanticError(
                    f"Semantic error: Field '{field_name}' not found in '{current_type_str}'.",
                    ctx.start.line,
                )

            field_info = struct_fields[field_name]
            field_idx = field_info["index"]
            current_type_str = field_info["type"]

            ptr = self.builder.gep(
                ptr,
                [ir.Constant(self.i32, 0), ir.Constant(self.i32, field_idx)],
                inbounds=True,
            )

        expr_ctx = ctx.expr()
        if expr_ctx is None:
            raise SemanticError(
                "Semantic error: Cannot recognize expression.", ctx.start.line
            )

        val = self.visit(expr_ctx)

        target_llvm_type = self.type_map[current_type_str]
        val = self.cast_to(val, target_llvm_type)  # pyright: ignore[reportArgumentType]

        _ = self.builder.store(val, ptr)
        return None

    @override
    def visitReadStmt(self, ctx: vibelangParser.ReadStmtContext) -> object:
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

        var_type = str(var_info["type"])

        if var_type == "float64":
            fmt_ptr = self.builder.bitcast(
                self.sfmt_float64, ir.IntType(8).as_pointer()
            )
        elif var_type == "float32":
            fmt_ptr = self.builder.bitcast(
                self.sfmt_float32, ir.IntType(8).as_pointer()
            )
        elif var_type == "int64":
            fmt_ptr = self.builder.bitcast(self.sfmt_int64, ir.IntType(8).as_pointer())
        elif var_type == "bool":
            tmp_ptr = self.builder.alloca(self.i32, name="bool_tmp_ptr")
            fmt_ptr = self.builder.bitcast(self.sfmt_bool, ir.IntType(8).as_pointer())
            _ = self.builder.call(self.scanf, [fmt_ptr, tmp_ptr])
            loaded_tmp = self.builder.load(tmp_ptr)

            bool_val = self.builder.icmp_unsigned(
                "!=", loaded_tmp, ir.Constant(self.i32, 0)
            )
            _ = self.builder.store(bool_val, ptr)
            return None
        else:  # int32
            fmt_ptr = self.builder.bitcast(self.sfmt_int32, ir.IntType(8).as_pointer())

        _ = self.builder.call(self.scanf, [fmt_ptr, ptr])
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
        elif val.type == self.i1:
            fmt_ptr = self.builder.bitcast(self.fmt_bool, ir.IntType(8).as_pointer())
            val = self.builder.zext(val, self.i32)
        else:  # i32
            fmt_ptr = self.builder.bitcast(self.fmt_int32, ir.IntType(8).as_pointer())

        _ = self.builder.call(self.printf, [fmt_ptr, val])
        return None

    @override
    def visitIfStmt(self, ctx: vibelangParser.IfStmtContext):
        """Generate an if / if-else statement using LLVM basic blocks."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        # Evaluate the condition expression and coerce it to an i1 (bool).
        expr_ctx = ctx.expr()
        if expr_ctx is None:
            raise SemanticError(
                "Semantic error: Cannot recognize expression.", ctx.start.line
            )

        cond_val = self.visit(expr_ctx)
        cond_val = self.to_bool(cond_val, ctx.start.line)  # pyright: ignore[reportArgumentType]

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Cannot allocate memory.", ctx.start.line
            )

        # The grammar flattens 'then' and 'else' statements into one child list,
        # so we walk the children and split them at the 'else' keyword.
        then_stmts = []
        else_stmts = []
        in_else = False

        for i in range(ctx.getChildCount()):
            child = ctx.getChild(i)
            if child.getText() == "else":
                in_else = (
                    True  # everything after this keyword belongs to the else-branch
                )
            elif isinstance(child, vibelangParser.StatementContext):  # pyright: ignore[reportUnnecessaryIsInstance]
                if in_else:
                    else_stmts.append(child)
                else:
                    then_stmts.append(child)

        has_else = in_else

        # Create the target blocks. The else block is only needed when an 'else' exists.
        then_block = self.builder.append_basic_block("if.then")
        else_block = self.builder.append_basic_block("if.else") if has_else else None
        merge_block = self.builder.append_basic_block("if.end")

        # Conditional jump: true -> then, false -> else (or straight to merge if no else).
        if has_else:
            _ = self.builder.cbranch(cond_val, then_block, else_block)
        else:
            _ = self.builder.cbranch(cond_val, then_block, merge_block)

        # --- THEN ---
        # New scope so variables declared inside the 'then' block don't leak out.
        self.builder.position_at_end(then_block)
        self.enter_scope()
        for stmt in then_stmts:
            self.visit(stmt)
        self.exit_scope()

        # Skip the branch if the block already ended with a return.
        if not self.builder.block.is_terminated:  # pyright: ignore[reportOptionalMemberAccess]
            _ = self.builder.branch(merge_block)

        # --- ELSE ---
        if has_else and else_block:
            self.builder.position_at_end(else_block)
            self.enter_scope()
            for stmt in else_stmts:
                self.visit(stmt)
            self.exit_scope()

            # Skip the branch if the block already ended with a return.
            if not self.builder.block.is_terminated:  # pyright: ignore[reportOptionalMemberAccess]
                _ = self.builder.branch(merge_block)

        # Continue emitting subsequent code in the merge block.
        self.builder.position_at_end(merge_block)

    @override
    def visitRelExpr(self, ctx: vibelangParser.RelExprContext):
        if self.builder is None:
            raise SemanticError(
                "Semantic error: Cannot allocate memory.", ctx.start.line
            )  # pyright: ignore[reportOptionalMemberAccess]

        left_val = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right_val = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]

        if left_val is None or right_val is None:
            raise SemanticError(
                "Semantic error: Invalid operand in relational expression.",
                ctx.start.line,
            )  # pyright: ignore[reportOptionalMemberAccess]

        left_val, right_val = self.promote_types(left_val, right_val)

        is_float = isinstance(left_val.type, (ir.FloatType, ir.DoubleType))

        operator = ctx.getChild(1).getText()

        try:
            if is_float:
                return self.builder.fcmp_ordered(
                    operator, left_val, right_val, name="rel_fcmp"
                )
            return self.builder.icmp_signed(
                operator, left_val, right_val, name="rel_icmp"
            )
        except ValueError:
            raise SemanticError(
                f"Semantic error: Unsupported relational operator '{operator}'.",
                ctx.start.line,
            )  # pyright: ignore[reportOptionalMemberAccess]

    @override
    def visitWhileStmt(self, ctx: vibelangParser.WhileStmtContext):
        """Generate a while loop using LLVM basic blocks."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Cannot allocate memory.", ctx.start.line
            )

        cond_block = self.builder.append_basic_block("while.cond")
        body_block = self.builder.append_basic_block("while.body")
        end_block = self.builder.append_basic_block("while.end")

        # Enter the loop by jumping to the condition check.
        _ = self.builder.branch(cond_block)

        # --- CONDITION ---
        # Evaluate the test each iteration: true -> body, false -> exit the loop.
        self.builder.position_at_end(cond_block)
        expr_ctx = ctx.expr()
        if expr_ctx is None:
            raise SemanticError(
                "Semantic error: Cannot recognize expression.", ctx.start.line
            )

        cond_val = self.visit(expr_ctx)
        cond_val = self.to_bool(cond_val, ctx.start.line)  # pyright: ignore[reportArgumentType]

        _ = self.builder.cbranch(cond_val, body_block, end_block)

        # --- BODY ---
        # Each iteration runs in its own scope so loop-local variables are fresh.
        self.builder.position_at_end(body_block)

        self.enter_scope()
        for stmt in ctx.statement():  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
            self.visit(stmt)
        self.exit_scope()

        # Loop back to re-check the condition, unless the body already terminated.
        if not self.builder.block.is_terminated:  # pyright: ignore[reportOptionalMemberAccess]
            _ = self.builder.branch(cond_block)

        # Continue emitting code after the loop.
        self.builder.position_at_end(end_block)

    def to_bool(self, val: ir.Value, line: int) -> ir.Value:
        """Helper converting other types to bool"""
        if val.type == self.i1:
            return val

        if isinstance(val.type, ir.IntType):
            return self.builder.icmp_unsigned(  # pyright: ignore[reportOptionalMemberAccess]
                "!=", val, ir.Constant(val.type, 0), name="tobool"
            )

        if isinstance(val.type, (ir.FloatType, ir.DoubleType)):
            return self.builder.fcmp_ordered(  # pyright: ignore[reportOptionalMemberAccess]
                "!=", val, ir.Constant(val.type, 0.0), name="tobool"
            )

        raise SemanticError("Semantic error: Cannot convert type to bool.", line)

    @override
    def visitBoolExpr(self, ctx: vibelangParser.BoolExprContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        val_str = ctx.BOOL().getText()  # pyright: ignore[reportOptionalMemberAccess]
        val = 1 if val_str == "true" else 0
        return ir.Constant(self.i1, val)

    @override
    def visitNotExpr(self, ctx: vibelangParser.NotExprContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        val = self.visit(ctx.expr())  # pyright: ignore[reportArgumentType]
        bool_val = self.to_bool(val, ctx.start.line)  # pyright: ignore[reportArgumentType]
        return self.builder.not_(bool_val, name="nottmp")  # pyright: ignore[reportOptionalMemberAccess]

    @override
    def visitAndExpr(self, ctx: vibelangParser.AndExprContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        left = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]

        left_bool = self.to_bool(left, ctx.start.line)  # pyright: ignore[reportArgumentType]
        right_bool = self.to_bool(right, ctx.start.line)  # pyright: ignore[reportArgumentType]

        return self.builder.and_(left_bool, right_bool, name="andtmp")  # pyright: ignore[reportOptionalMemberAccess]

    @override
    def visitOrExpr(self, ctx: vibelangParser.OrExprContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        left = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]

        left_bool = self.to_bool(left, ctx.start.line)  # pyright: ignore[reportArgumentType]
        right_bool = self.to_bool(right, ctx.start.line)  # pyright: ignore[reportArgumentType]

        return self.builder.or_(left_bool, right_bool, name="ortmp")  # pyright: ignore[reportOptionalMemberAccess]

    @override
    def visitXorExpr(self, ctx: vibelangParser.XorExprContext) -> object:
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        left = self.visit(ctx.expr(0))  # pyright: ignore[reportArgumentType]
        right = self.visit(ctx.expr(1))  # pyright: ignore[reportArgumentType]

        left_bool = self.to_bool(left, ctx.start.line)  # pyright: ignore[reportArgumentType]
        right_bool = self.to_bool(right, ctx.start.line)  # pyright: ignore[reportArgumentType]

        return self.builder.xor(left_bool, right_bool, name="xortmp")  # pyright: ignore[reportOptionalMemberAccess]

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

    @override
    def visitFunctionDefinition(
        self, ctx: vibelangParser.FunctionDefinitionContext
    ) -> object:
        """Define a top-level function and emit its body."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        func_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

        # Resolve the return type.
        return_type_ctx = ctx.returnType()
        if return_type_ctx is None:
            raise SemanticError(
                "Semantic error: Cannot recognize return type.", ctx.start.line
            )

        ret_type_str = return_type_ctx.getText()
        if ret_type_str == "void":
            ret_llvm_type = ir.VoidType()
        else:
            if ret_type_str not in self.type_map:
                raise SemanticError(
                    f"Semantic error: Unsupported return type '{ret_type_str}'.",
                    ctx.start.line,
                )
            ret_llvm_type = self.type_map[ret_type_str]

        # Collect the parameter types/names so we can build the function signature.
        param_llvm_types = []
        param_names = []
        param_type_strs = []

        params_ctx = ctx.parameters()
        if params_ctx is not None:
            for p_ctx in params_ctx.parameter():  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
                p_type_str = p_ctx.type_().getText()
                p_name = p_ctx.ID().getText()

                if p_type_str not in self.type_map:
                    raise SemanticError(
                        f"Semantic error: Unsupported parameter type '{p_type_str}'.",
                        ctx.start.line,
                    )

                param_llvm_types.append(self.type_map[p_type_str])
                param_names.append(p_name)
                param_type_strs.append(p_type_str)

        # Build the signature and declare the function (reject redefinitions).
        func_type = ir.FunctionType(ret_llvm_type, param_llvm_types)

        if func_name in self.module.globals:
            raise SemanticError(
                f"Semantic error: Function '{func_name}' is already defined.",
                ctx.start.line,
            )

        func = ir.Function(self.module, func_type, name=func_name)

        # Name the IR arguments to match the source (purely for readable IR).
        for arg, name in zip(func.args, param_names):
            arg.name = name

        # Remember the builder of the enclosing context (usually 'main') so we
        # can restore it after we finish emitting this function's body.
        saved_builder = self.builder

        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        # Fresh scope for the function body; track the expected return type so
        # 'return' statements can validate/cast their value.
        self.enter_scope()

        self.current_function_return_type = ret_llvm_type

        # Parameters are passed by value, so copy each one into a local alloca.
        # This lets the body treat parameters like ordinary mutable variables.
        for arg, name, type_str in zip(func.args, param_names, param_type_strs):
            ptr = self.allocate_variable(name, type_str, ctx.start.line)
            _ = self.builder.store(arg, ptr)

        for stmt in ctx.statement():  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
            self.visit(stmt)

        # If the body falls off the end without an explicit 'return', emit a
        # default terminator so the IR is valid (void -> ret void, otherwise 0).
        if not self.builder.block.is_terminated:  # pyright: ignore[reportOptionalMemberAccess]
            if isinstance(ret_llvm_type, ir.VoidType):
                _ = self.builder.ret_void()
            elif isinstance(ret_llvm_type, (ir.FloatType, ir.DoubleType)):
                _ = self.builder.ret(ir.Constant(ret_llvm_type, 0.0))
            else:
                _ = self.builder.ret(ir.Constant(ret_llvm_type, 0))

        # Tear down the function context and restore the previous builder/scope.
        self.current_function_return_type = None

        self.exit_scope()
        self.builder = saved_builder

        return None

    @override
    def visitReturnStmt(self, ctx: vibelangParser.ReturnStmtContext) -> object:
        """Emit a 'return' statement."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        # current_function_return_type is only set inside a function body.
        if self.current_function_return_type is None:
            raise SemanticError(
                "Semantic error: 'return' statement is only allowed inside functions.",
                ctx.start.line,
            )

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        expr_ctx = ctx.expr()

        # void function: 'return;' is allowed, 'return value;' is not.
        if isinstance(self.current_function_return_type, ir.VoidType):
            if expr_ctx is not None:
                raise SemanticError(
                    "Semantic error: Void function cannot return a value.",
                    ctx.start.line,
                )
            _ = self.builder.ret_void()
            return None

        # non-void function: a value is required.
        if expr_ctx is None:
            raise SemanticError(
                "Semantic error: Expected an expression to return.", ctx.start.line
            )

        # Cast the value to the declared return type before returning it.
        val = self.visit(expr_ctx)
        val = self.cast_to(val, self.current_function_return_type)  # pyright: ignore[reportArgumentType]

        _ = self.builder.ret(val)
        return None

    @override
    def visitCallExpr(self, ctx: vibelangParser.CallExprContext) -> object:
        """Call a function and yield its result as an expression value."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        # The callee must be a previously declared function in the module.
        func_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
        if func_name not in self.module.globals:
            raise SemanticError(
                f"Semantic error: Undefined function '{func_name}'.", ctx.start.line
            )

        func = self.module.globals[func_name]
        if not isinstance(func, ir.Function):
            raise SemanticError(
                f"Semantic error: '{func_name}' is not a function.", ctx.start.line
            )

        # Evaluate and type-adjust the arguments; arity must match exactly.
        args_vals = []
        args_ctx = ctx.arguments()

        expected_arg_count = len(func.args)

        if args_ctx is not None:
            exprs = args_ctx.expr()
            provided_arg_count = len(exprs)  # pyright: ignore[reportArgumentType]

            if expected_arg_count != provided_arg_count:
                raise SemanticError(
                    f"Semantic error: Function '{func_name}' expects {expected_arg_count} arguments, got {provided_arg_count}.",
                    ctx.start.line,
                )

            for i, expr_ctx in enumerate(exprs):  # pyright: ignore[reportArgumentType]
                val = self.visit(expr_ctx)
                expected_type = func.args[i].type
                val = self.cast_to(
                    val, expected_type
                )  # implicit cast to the parameter type  # pyright: ignore[reportArgumentType]
                args_vals.append(val)
        elif expected_arg_count > 0:
            raise SemanticError(
                f"Semantic error: Function '{func_name}' expects {expected_arg_count} arguments, got 0.",
                ctx.start.line,
            )

        # A void call cannot be given a result name in the IR.
        is_void = isinstance(func.function_type.return_type, ir.VoidType)
        call_name = "" if is_void else f"{func_name}_res"

        return self.builder.call(func, args_vals, name=call_name)

    @override
    def visitCallStmt(self, ctx: vibelangParser.CallStmtContext) -> object:
        """Call a function as a statement, discarding any return value."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        func_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
        if func_name not in self.module.globals:
            raise SemanticError(
                f"Semantic error: Undefined function '{func_name}'.", ctx.start.line
            )

        func = self.module.globals[func_name]
        if not isinstance(func, ir.Function):
            raise SemanticError(
                f"Semantic error: '{func_name}' is not a function.", ctx.start.line
            )

        # Same arity check + per-argument casting as visitCallExpr.
        args_vals = []
        args_ctx = ctx.arguments()

        expected_arg_count = len(func.args)

        if args_ctx is not None:
            exprs = args_ctx.expr()
            provided_arg_count = len(exprs)  # pyright: ignore[reportArgumentType]

            if expected_arg_count != provided_arg_count:
                raise SemanticError(
                    f"Semantic error: Function '{func_name}' expects {expected_arg_count} arguments, got {provided_arg_count}.",
                    ctx.start.line,
                )

            for i, expr_ctx in enumerate(exprs):  # pyright: ignore[reportArgumentType]
                val = self.visit(expr_ctx)
                expected_type = func.args[i].type
                val = self.cast_to(val, expected_type)  # pyright: ignore[reportArgumentType]
                args_vals.append(val)
        elif expected_arg_count > 0:
            raise SemanticError(
                f"Semantic error: Function '{func_name}' expects {expected_arg_count} arguments, got 0.",
                ctx.start.line,
            )

        # Result intentionally ignored.
        _ = self.builder.call(func, args_vals)
        return None

    @override
    def visitStructDefinition(
        self, ctx: vibelangParser.StructDefinitionContext
    ) -> object:
        """Register a struct type from its field declarations."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        struct_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

        if struct_name in self.type_map:
            raise SemanticError(
                f"Semantic error: Type '{struct_name}' is already defined.",
                ctx.start.line,
            )

        # Create a named LLVM struct; its body is filled in once fields are known.
        llvm_struct = self.module.context.get_identified_type(struct_name)

        fields_metadata = {}
        field_llvm_types = []

        # Field order matters: the index 'i' is the field's position inside the
        # struct and is reused later for GEP (address-of) and extract_value.
        for i, field_ctx in enumerate(ctx.structField()):  # pyright: ignore[reportArgumentType]
            field_type_str = field_ctx.type_().getText()
            field_name = field_ctx.ID().getText()

            if field_type_str not in self.type_map:
                raise SemanticError(
                    f"Semantic error: Unknown field type '{field_type_str}'.",
                    ctx.start.line,
                )

            fields_metadata[field_name] = {"index": i, "type": field_type_str}
            field_llvm_types.append(self.type_map[field_type_str])

        # Set the struct layout and make the type usable like any other type.
        _ = llvm_struct.set_body(*field_llvm_types)

        self.type_map[struct_name] = llvm_struct
        self.struct_info[struct_name] = {"type": llvm_struct, "fields": fields_metadata}

        return None

    @override
    def visitStructInitExpr(self, ctx: vibelangParser.StructInitExprContext) -> object:
        """Build a struct value from a brace initializer."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        struct_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
        if struct_name not in self.struct_info:
            raise SemanticError(
                f"Semantic error: Unknown struct/class '{struct_name}'.", ctx.start.line
            )

        struct_def = cast("dict", self.struct_info[struct_name])  # pyright: ignore[reportMissingTypeArgument]
        struct_type = cast("ir.Type", struct_def["type"])
        struct_fields = cast("dict", struct_def["fields"])  # pyright: ignore[reportMissingTypeArgument]

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        # Temporary stack slot we fill field-by-field.
        ptr = self.builder.alloca(struct_type, name=f"{struct_name}_init")

        field_init_list = ctx.fieldInitList()
        if field_init_list is not None:
            ids = field_init_list.ID()  # field names in 'name: value' pairs
            exprs = field_init_list.expr()  # matching value expressions

            for i, id_node in enumerate(ids):  # pyright: ignore[reportArgumentType]
                field_name = id_node.getText()

                if field_name not in struct_fields:
                    raise SemanticError(
                        f"Semantic error: Field '{field_name}' not in '{struct_name}'.",
                        ctx.start.line,
                    )

                field_info = struct_fields[field_name]
                field_idx = field_info["index"]
                field_target_type_str = field_info["type"]

                val = self.visit(exprs[i])  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]
                val = self.cast_to(val, self.type_map[field_target_type_str])  # pyright: ignore[reportArgumentType]

                # GEP with indices [0, field_idx] = address of that field inside
                # the struct; the leading 0 dereferences the pointer itself.
                field_ptr = self.builder.gep(
                    ptr,
                    [ir.Constant(self.i32, 0), ir.Constant(self.i32, field_idx)],
                    inbounds=True,
                )
                _ = self.builder.store(val, field_ptr)

        # Return the struct by value (loaded out of the temporary slot).
        return self.builder.load(ptr)

    @override
    def visitMemberAccessExpr(
        self, ctx: vibelangParser.MemberAccessExprContext
    ) -> object:
        """Read a field out of a struct value: expr '.' ID."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        # Evaluate the base expression; it must be a (named) struct value.
        struct_val = cast("ir.Value", self.visit(ctx.expr()))  # pyright: ignore[reportInvalidCast, reportArgumentType]
        field_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

        struct_type = struct_val.type

        if not isinstance(struct_type, ir.IdentifiedStructType):
            raise SemanticError(
                "Semantic error: Attempted member access on a non-struct type.",
                ctx.start.line,
            )

        struct_name = struct_type.name
        struct_def = cast("dict", self.struct_info[struct_name])  # pyright: ignore[reportMissingTypeArgument]
        struct_fields = cast("dict", struct_def["fields"])  # pyright: ignore[reportMissingTypeArgument]

        if field_name not in struct_fields:
            raise SemanticError(
                f"Semantic error: Field '{field_name}' does not exist on '{struct_name}'.",
                ctx.start.line,
            )

        field_idx = struct_fields[field_name]["index"]

        # extract_value pulls one element out of an aggregate value by index.
        return self.builder.extract_value(
            struct_val, field_idx, name=f"extract_{field_name}"
        )

    @override
    def visitClassDefinition(
        self, ctx: vibelangParser.ClassDefinitionContext
    ) -> object:
        """Define a class: its data layout plus its methods."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        class_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
        if class_name in self.type_map:
            raise SemanticError(
                f"Semantic error: Type '{class_name}' is already defined.",
                ctx.start.line,
            )

        # Like a struct, the class gets a named LLVM type for its fields.
        llvm_class = self.module.context.get_identified_type(class_name)

        fields_metadata = {}
        field_llvm_types = []
        methods_ctx = []

        # First pass: lay out the fields and remember the methods for later.
        # Fields and methods can be interleaved in the source, so we sort them.
        for member_ctx in ctx.classMember():  # pyright: ignore[reportOptionalIterable, reportGeneralTypeIssues]
            if isinstance(
                member_ctx, vibelangParser.ClassFieldContext
            ):  # a field (data member)
                field_type_str = member_ctx.type_().getText()  # pyright: ignore[reportOptionalMemberAccess]
                field_name = member_ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

                if field_type_str not in self.type_map:
                    raise SemanticError(
                        f"Semantic error: Unknown field type '{field_type_str}'.",
                        ctx.start.line,
                    )

                # index = current field count, i.e. its position in the struct.
                fields_metadata[field_name] = {
                    "index": len(field_llvm_types),
                    "type": field_type_str,
                }
                field_llvm_types.append(self.type_map[field_type_str])

            elif isinstance(member_ctx, vibelangParser.ClassMethodContext):  # a method
                methods_ctx.append(member_ctx)

        # Finalize the layout and register the type. We store the layout in BOTH
        # struct_info (so field access/GEP works) and class_info (so method
        # lookup and 'new' know this is a class).
        _ = llvm_class.set_body(*field_llvm_types)

        self.type_map[class_name] = llvm_class

        self.struct_info[class_name] = {"type": llvm_class, "fields": fields_metadata}
        self.class_info[class_name] = {"type": llvm_class, "fields": fields_metadata}

        # Second pass: emit each method as a standalone function.
        for m_ctx in methods_ctx:
            method_name = m_ctx.ID().getText()
            full_method_name = f"{class_name}_{method_name}"  # name-mangled global

            ret_type_str = m_ctx.returnType().getText()
            if ret_type_str == "void":
                ret_llvm_type = ir.VoidType()
            else:
                ret_llvm_type = self.type_map[ret_type_str]

            # The first parameter is always 'self': a pointer to the instance.
            # Passing a pointer (not a value) is what lets methods mutate fields.
            param_llvm_types = [llvm_class.as_pointer()]
            param_names = ["self"]
            param_type_strs = [class_name]

            # Append any explicit parameters declared in the source.
            params_ctx = m_ctx.parameters()
            if params_ctx is not None:
                for p_ctx in params_ctx.parameter():
                    p_type_str = p_ctx.type_().getText()
                    p_name = p_ctx.ID().getText()
                    param_llvm_types.append(self.type_map[p_type_str])
                    param_names.append(p_name)
                    param_type_strs.append(p_type_str)

            func_type = ir.FunctionType(ret_llvm_type, param_llvm_types)
            func = ir.Function(self.module, func_type, name=full_method_name)

            for arg, name in zip(func.args, param_names):
                arg.name = name

            # Switch the builder into this method's body (restored afterwards).
            saved_builder = self.builder
            block = func.append_basic_block(name="entry")
            self.builder = ir.IRBuilder(block)

            self.enter_scope()
            self.current_function_return_type = ret_llvm_type

            # Bind parameters in the method scope. 'self' is special: it is kept
            # as the raw pointer (no alloca/copy) so writes go to the real object;
            # ordinary parameters are copied into local stack slots as usual.
            for i, (arg, name, type_str) in enumerate(
                zip(func.args, param_names, param_type_strs)
            ):
                if i == 0:
                    self.current_scope["self"] = {"ptr": arg, "type": class_name}
                else:
                    ptr = self.allocate_variable(name, type_str, m_ctx.start.line)
                    _ = self.builder.store(arg, ptr)

            for stmt in m_ctx.statement():
                self.visit(stmt)

            # Provide a default terminator if the method body doesn't return.
            if not self.builder.block.is_terminated:  # pyright: ignore[reportOptionalMemberAccess]
                if isinstance(ret_llvm_type, ir.VoidType):
                    _ = self.builder.ret_void()
                elif isinstance(ret_llvm_type, (ir.FloatType, ir.DoubleType)):
                    _ = self.builder.ret(ir.Constant(ret_llvm_type, 0.0))
                else:
                    _ = self.builder.ret(ir.Constant(ret_llvm_type, 0))

            # Restore the outer context before emitting the next method.
            self.current_function_return_type = None
            self.exit_scope()
            self.builder = saved_builder

        return None

    @override
    def visitNewObjectExpr(self, ctx: vibelangParser.NewObjectExprContext) -> object:
        """Allocate a new object instance: 'new' ID '(' arguments? ')'."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        class_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
        if class_name not in self.class_info:
            raise SemanticError(
                f"Semantic error: Unknown class '{class_name}'.", ctx.start.line
            )

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        llvm_type = self.type_map[class_name]

        # Reserve space for the instance and return it as a value.
        ptr = self.builder.alloca(llvm_type, name=f"new_{class_name}")
        return self.builder.load(ptr)

    @override
    def visitSelfExpr(self, ctx: vibelangParser.SelfExprContext) -> object:
        """Resolve the 'self' keyword inside a method to the current instance."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")

        var_info = self.lookup_variable("self", ctx.start.line)
        ptr = cast("ir.Value", var_info["ptr"])
        typ = self.type_map[var_info["type"]]  # pyright: ignore[reportArgumentType]

        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        return self.builder.load(ptr, typ=typ, name="self_val")

    def get_pointer_and_class(self, ctx) -> tuple[ir.Value, str]:  # pyright: ignore[reportUnknownParameterType, reportMissingParameterType]
        """Resolve an expression to an *address* (pointer) and its class/struct name."""
        if isinstance(ctx, vibelangParser.IdExprContext):
            var_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]
            var_info = self.lookup_variable(var_name, ctx.start.line)  # pyright: ignore[reportOptionalMemberAccess]
            return cast("ir.Value", var_info["ptr"]), str(var_info["type"])

        if isinstance(ctx, vibelangParser.SelfExprContext):
            var_info = self.lookup_variable("self", ctx.start.line)  # pyright: ignore[reportOptionalMemberAccess]
            return cast("ir.Value", var_info["ptr"]), str(var_info["type"])

        if isinstance(ctx, vibelangParser.MemberAccessExprContext):
            # Recurse to the base's pointer, then index into the requested field.
            base_ptr, base_class_name = self.get_pointer_and_class(ctx.expr())
            field_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

            struct_def = cast("dict", self.struct_info[base_class_name])  # pyright: ignore[reportMissingTypeArgument]
            struct_fields = cast("dict", struct_def["fields"])  # pyright: ignore[reportMissingTypeArgument]

            if field_name not in struct_fields:
                raise SemanticError(
                    f"Semantic error: Field '{field_name}' not found.", ctx.start.line
                )  # pyright: ignore[reportOptionalMemberAccess]

            field_info = struct_fields[field_name]
            field_idx = field_info["index"]
            field_type_str = field_info["type"]

            # Address of base[field_idx]; the returned pointer can be stored into.
            new_ptr = self.builder.gep(  # pyright: ignore[reportOptionalMemberAccess]
                base_ptr,
                [ir.Constant(self.i32, 0), ir.Constant(self.i32, field_idx)],
                inbounds=True,
            )
            return new_ptr, field_type_str

        # Not an addressable expression (e.g. a literal or a call result).
        raise ValueError("Not an L-value")

    def handle_method_call(
        self,
        ctx: vibelangParser.MethodCallExprContext
        | vibelangParser.MethodCallStmtContext,
    ) -> object:
        """Shared logic for method calls, used by both the expression and statement forms."""
        if ctx.start is None:
            raise SemanticError("Semantic error: Cannot recognize line number.")
        if self.builder is None:
            raise SemanticError(
                "Semantic error: Builder is not initialized.", ctx.start.line
            )

        expr_ctx = ctx.expr()
        method_name = ctx.ID().getText()  # pyright: ignore[reportOptionalMemberAccess]

        # Preferred path: the receiver is addressable (a variable, self, or a
        # field), so we call the method on the real object's pointer.
        try:
            obj_ptr, class_name = self.get_pointer_and_class(expr_ctx)
        except ValueError:
            # Fallback: the receiver is a temporary value (e.g. a call result).
            # Identify its class by matching the LLVM type, then spill it to a
            # stack slot so we still have a pointer to pass as 'self'.
            obj_val = cast("ir.Value", self.visit(expr_ctx))  # pyright: ignore[reportInvalidCast, reportArgumentType]
            class_name = None
            for name, llvm_typ in self.type_map.items():
                if llvm_typ == obj_val.type:
                    class_name = name
                    break
            if class_name is None or class_name not in self.class_info:
                raise SemanticError(
                    "Semantic error: Cannot call method on non-object.", ctx.start.line
                )

            obj_ptr = self.builder.alloca(obj_val.type, name="temp_obj_ptr")
            _ = self.builder.store(obj_val, obj_ptr)

        if class_name not in self.class_info:
            raise SemanticError(
                f"Semantic error: Type '{class_name}' is not a class.", ctx.start.line
            )

        # Methods are stored under their mangled name '<Class>_<method>'.
        full_func_name = f"{class_name}_{method_name}"
        if full_func_name not in self.module.globals:
            raise SemanticError(
                f"Semantic error: Method '{method_name}' not found in class '{class_name}'.",
                ctx.start.line,
            )

        func = self.module.globals[full_func_name]

        # The instance pointer is always the first argument ('self').
        args_vals = [obj_ptr]
        args_ctx = ctx.arguments()

        # -1 because 'self' is implicit and not written at the call site.
        expected_arg_count = len(func.args) - 1

        if args_ctx is not None:
            exprs = args_ctx.expr()
            provided_arg_count = len(exprs)  # pyright: ignore[reportArgumentType]

            if expected_arg_count != provided_arg_count:
                raise SemanticError(
                    f"Semantic error: Method '{method_name}' expects {expected_arg_count} arguments, got {provided_arg_count}.",
                    ctx.start.line,
                )

            for i, arg_expr in enumerate(exprs):  # pyright: ignore[reportArgumentType]
                val = cast("ir.Value", self.visit(arg_expr))  # pyright: ignore[reportInvalidCast]
                expected_type = func.args[i + 1].type  # +1 to skip 'self'
                val = self.cast_to(val, expected_type)
                args_vals.append(val)
        elif expected_arg_count > 0:
            raise SemanticError(
                f"Semantic error: Method '{method_name}' expects {expected_arg_count} arguments, got 0.",
                ctx.start.line,
            )

        # void calls can't be named in the IR.
        is_void = isinstance(func.function_type.return_type, ir.VoidType)
        call_name = "" if is_void else f"{method_name}_res"

        return self.builder.call(func, args_vals, name=call_name)

    @override
    def visitMethodCallExpr(self, ctx: vibelangParser.MethodCallExprContext) -> object:
        """Method call used as an expression (its return value is used)."""
        return self.handle_method_call(ctx)

    @override
    def visitMethodCallStmt(self, ctx: vibelangParser.MethodCallStmtContext) -> object:
        """Method call used as a statement (the return value is discarded)."""
        _ = self.handle_method_call(ctx)
        return None
