import re
import ast
from typing import Any, List, Union
from collections import defaultdict, deque


class APIMatcher:
    def __init__(self, code, apis, maxline=1000):
        self.code = code
        self.apis = apis
        self.matched_funcs = []
        self.maxline = maxline
    

    def match(self):
        try:
            self.tree: ast.Module = ast.parse(self.code, mode="exec")
        except SyntaxError:
            return self
        self.solve_imports()
        self.solve_alias()
        self.splitlines_no_ff()
        self.extract_global_ref_dict()

        self.match_script_stmts()

        for node in self.tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.match_one_func(node, self.global_ref_dict)
            elif isinstance(node, ast.ClassDef):
                class_ref_dict = self.global_ref_dict.copy()
                class_ref_dict.update(self.extract_class_ref_dict(node))
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.match_one_func(member, class_ref_dict)
        return self
            
            
    def solve_imports(self):
        import_dict = dict()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for nn in node.names: 
                    if nn.asname is None:
                        import_dict[nn.name] = nn.name
                    else:
                        import_dict[nn.asname] = nn.name 
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                for nn in node.names: 
                    if nn.asname is None: # alias name not found
                        if nn.name == "*":
                            import_dict[node.module] = node.module
                        else:
                            import_dict[nn.name] = node.module + '.' + nn.name
                    else:
                        import_dict[nn.asname] = node.module + '.' + nn.name
        # prefixes = {api.split(".")[0] for api in self.apis}
        # import_dict = {k:v for k, v in import_dict.items() if v.split(".")[0] in prefixes}
        self.import_dict = import_dict

    
    def solve_alias(self):
        prefix2apis = defaultdict(set)
        for api in self.apis:
            parts = api.split(".")
            for i in range(1, len(parts), 1):
                prefix = ".".join(parts[:i])
                prefix2apis[prefix].add(api)

        alias_dict = dict()
        for surface_name, actual_name in self.import_dict.items():
            if actual_name in self.apis:    # a target API is directly imported
                alias_dict[surface_name] = actual_name
            elif actual_name in prefix2apis: 
                for api in prefix2apis[actual_name]:
                    surface_api_name = surface_name + api.replace(actual_name, "")
                    alias_dict[surface_api_name] = api
        self.alias_dict = alias_dict

    
    def match_one_func(self, func: Union[ast.FunctionDef,ast.AsyncFunctionDef], ref_dict: dict):
        if func.end_lineno - func.lineno > self.maxline:
            return
        ref_dict = ref_dict.copy() if ref_dict is not None else dict()
        for node in ast.walk(func):
            if not isinstance(node, ast.Call):
                continue
            if not (func.lineno <= node.lineno <= func.end_lineno):
                continue
            
            call_name = self.get_source_by_node(node.func)
            
            ref_dict.update(self.extract_local_ref_dict(func))
            resolved_call_name = call_name
            name_parts = resolved_call_name.split('.')
            if len(name_parts) >= 2:
                qualifier = ".".join(name_parts[:-1])
                if qualifier in ref_dict:
                    resolved_call_name = f"{ref_dict[qualifier]}.{name_parts[-1]}"
            resolved_call_name = self.alias_dict.get(resolved_call_name, resolved_call_name)
            if resolved_call_name not in self.apis:
                continue

            func_source = self.get_source_by_node(func)
            self.matched_funcs.append({
                "function": " " * func.col_offset + func_source,
                "import dict": self.import_dict,
                "alias dict": self.alias_dict,
                "reference dict": ref_dict,
                "matched call": {
                    "call name": call_name,
                    "position": (node.lineno - func.lineno, node.col_offset),
                    "matched api": resolved_call_name
                }
            })
            return
        

    def match_script_stmts(self, max_stmts=5):
        block = []
        for stmt in self.tree.body:
            if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
                block = []
                continue
            block.append(stmt)
            if len(block) > max_stmts:
                block = block[-max_stmts:]
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                if len(block) == 1:
                    continue
                call_name = self.get_source_by_node(node.func)
                resolved_call_name = call_name
                name_parts = resolved_call_name.split('.')
                if len(name_parts) >= 2:
                    qualifier = ".".join(name_parts[:-1])
                    if qualifier in self.global_ref_dict:
                        resolved_call_name = f"{self.global_ref_dict[qualifier]}.{name_parts[-1]}"
                resolved_call_name = self.alias_dict.get(resolved_call_name, resolved_call_name)
                if resolved_call_name not in self.apis:
                    continue
                
                lineno = block[0].lineno
                col_offset = block[0].col_offset
                end_lineno = block[-1].end_lineno
                end_col_offset = block[-1].end_col_offset

                source = self.get_source_by_position(lineno - 1, col_offset, end_lineno - 1, end_col_offset)
                self.matched_funcs.append({
                    "function": " " * col_offset + source,
                    "import dict": self.import_dict,
                    "alias dict": self.alias_dict,
                    "reference dict": self.global_ref_dict,
                    "matched call": {
                        "call name": call_name,
                        "position": (node.lineno - lineno, node.col_offset),
                        "matched api": resolved_call_name
                    }
                })
                return
        

    def extract_global_ref_dict(self):
        '''TODO: perform type inference to get a more accurate reference dict'''
        ref_dict = {}
        for node in self.tree.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call_name = self.get_source_by_node(node.value.func)
            assign_name = self.get_source_by_node(node.targets[0])
            if not assign_name or not call_name:
                continue
            call_parts = call_name.split(".")
            if call_parts[-1][0].isupper():
                ref_dict[assign_name] = call_name
            elif len(call_parts) >= 2 and call_parts[-1] == "from_pretrained": # for transformers
                ref_dict[assign_name] = ".".join(call_parts[:-1])
        self.global_ref_dict = ref_dict
    

    def extract_class_ref_dict(self, clazz: ast.ClassDef):
        '''TODO: perform type inference to get a more accurate reference dict'''
        ref_dict = {}
        for node in clazz.body:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call_name = self.get_source_by_node(node.value.func)
            assign_name = self.get_source_by_node(node.targets[0])
            if assign_name and call_name and call_name.split(".")[-1][0].isupper():
                ref_dict[f"self.{assign_name}"] = call_name
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                ref_dict.update(self.extract_local_ref_dict(node))
        return ref_dict
    

    def extract_local_ref_dict(self, func: ast.FunctionDef):
        '''TODO: perform type inference to get a more accurate reference dict'''
        ref_dict = {}
        for node in ast.walk(func):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            call_name = self.get_source_by_node(node.value.func)
            assign_name = self.get_source_by_node(node.targets[0])
            if not assign_name or not call_name:
                continue
            call_parts = call_name.split(".")
            if call_parts[-1][0].isupper():
                ref_dict[assign_name] = call_name
            elif len(call_parts) >= 2 and call_parts[-1] == "from_pretrained": # for transformers
                ref_dict[assign_name] = ".".join(call_parts[:-1])
        return ref_dict
    

    def splitlines_no_ff(self):
        """Split a string into lines ignoring form feed and other chars.
        This mimics how the Python parser splits source code.
        """
        idx = 0
        lines = []
        next_line = ''
        while idx < len(self.code):
            c = self.code[idx]
            next_line += c
            idx += 1
            # Keep \r\n together
            if c == '\r' and idx < len(self.code) and self.code[idx] == '\n':
                next_line += '\n'
                idx += 1
            if c in '\r\n':
                lines.append(next_line)
                next_line = ''

        if next_line:
            lines.append(next_line)
        self.lines = lines
    

    def get_source_by_node(self, node: ast.AST):
        try:
            if node.end_lineno is None or node.end_col_offset is None:
                return ""
            lineno = node.lineno - 1
            end_lineno = node.end_lineno - 1
            col_offset = node.col_offset
            end_col_offset = node.end_col_offset
        except AttributeError:
            return ""
        
        return self.get_source_by_position(lineno, col_offset, end_lineno, end_col_offset)


    def get_source_by_position(self, lineno, col_offset, end_lineno, end_col_offset):
        if end_lineno == lineno:
            return self.lines[lineno].encode()[col_offset:end_col_offset].decode()

        first = self.lines[lineno].encode()[col_offset:].decode()
        last = self.lines[end_lineno].encode()[:end_col_offset].decode()
        lines = self.lines[lineno+1:end_lineno]
        return f"{first}{''.join(lines)}{last}"

    # def resolve_name(self, name: ast.expr):
    #     if isinstance(name, ast.Name):
    #         return name.id
    #     elif isinstance(name, ast.Attribute):
    #         attr = name.attr
    #         qualifier = name.value
    #         qualifiers = []
    #         while isinstance(qualifier, ast.Attribute):
    #             qualifiers.insert(0, qualifier.attr)
    #             qualifier = qualifier.value
    #         if isinstance(qualifier, ast.Name):
    #             qualifiers.insert(0, qualifier.id)
    #         elif isinstance(qualifier, ast.Constant):
    #             qualifiers.insert(0, type(qualifier.value).__name__)
    #         else:
    #             resolved_q =  ast.get_source_segment(self.code, qualifier)
    #             if "torch.linalg.norm" in resolved_q:
    #                 print(f"attr: {qualifier}")
    #                 print(f"xxx: {ast.get_source_segment(self.code, name)}")
    #             qualifiers.insert(0, resolved_q)
    #         return f"{'.'.join(qualifiers)}.{attr}"
    #     else:
    #         resolved_name = ast.get_source_segment(self.code, name)
    #         if "torch.linalg.norm" in resolved_name:
    #             print(f"name: {resolved_name}")
    #         return resolved_name




