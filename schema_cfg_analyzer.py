"""
Empirical analysis pipeline for:
  "Validation-Gated Graph Reachability" (Alpay, 2026)

Loads JSONSchemaBench (10K real-world JSON schemas),
converts each to a CFG, classifies linear vs general,
and outputs statistics for the manuscript's empirical section.

Usage:
  pip install datasets jsonschema
  python schema_cfg_analyzer.py
"""

import json
import sys
import csv
import os
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, defaultdict


@dataclass(frozen=True)
class Terminal:
    value: str
    def __repr__(self): return f"'{self.value}'"

@dataclass(frozen=True)
class Nonterminal:
    name: str
    def __repr__(self): return self.name

@dataclass
class Production:
    lhs: str
    rhs: list  # list of Terminal | Nonterminal

    @property
    def rhs_nonterminals(self):
        return [s for s in self.rhs if isinstance(s, Nonterminal)]

    @property
    def rhs_nt_count(self):
        return len(self.rhs_nonterminals)

    @property
    def is_linear(self):
        return self.rhs_nt_count <= 1

@dataclass
class CFG:
    start: str
    productions: list = field(default_factory=list)
    nonterminals: set = field(default_factory=set)
    terminals: set = field(default_factory=set)

    @property
    def P(self): return len(self.productions)

    @property
    def N(self): return len(self.nonterminals)

    @property
    def is_linear(self):
        return all(p.is_linear for p in self.productions)

    @property
    def max_rhs_nt(self):
        if not self.productions: return 0
        return max(p.rhs_nt_count for p in self.productions)

    @property
    def nonlinear_count(self):
        return sum(1 for p in self.productions if not p.is_linear)

    def add(self, lhs, rhs):
        self.nonterminals.add(lhs)
        for s in rhs:
            if isinstance(s, Nonterminal):
                self.nonterminals.add(s.name)
            elif isinstance(s, Terminal):
                self.terminals.add(s.value)
        self.productions.append(Production(lhs, rhs))



class SchemaConverter:
    """
    Converts a JSON Schema to a CFG at *structural* level.
    Terminals = JSON tokens (braces, brackets, colons, commas, literals).
    Nonterminals = schema-derived type/structure nodes.
    """

    def __init__(self):
        self.cfg = CFG(start="S")
        self._counter = 0
        self._ref_cache = {}
        self._root_schema = None
        self._depth = 0
        self._max_depth = 30  # recursion guard

    def fresh(self, prefix="N"):
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def convert(self, schema: dict) -> Optional[CFG]:
        if not isinstance(schema, dict):
            return None
        self._root_schema = schema
        try:
            nt = self._convert_node(schema)
            if nt:
                self.cfg.add("S", [Nonterminal(nt)])
            return self.cfg
        except (RecursionError, KeyError, TypeError, ValueError):
            return self.cfg if self.cfg.productions else None

    def _convert_node(self, node) -> Optional[str]:
        if not isinstance(node, dict):
            return None

        self._depth += 1
        if self._depth > self._max_depth:
            self._depth -= 1
            nt = self.fresh("TRUNC")
            self.cfg.add(nt, [Terminal("<truncated>")])
            return nt

        try:
            if "$ref" in node:
                return self._resolve_ref(node["$ref"])

            if "const" in node:
                nt = self.fresh("CONST")
                self.cfg.add(nt, [Terminal(json.dumps(node["const"]))])
                return nt

            if "enum" in node:
                nt = self.fresh("ENUM")
                for val in node["enum"]:
                    self.cfg.add(nt, [Terminal(json.dumps(val))])
                return nt

            if "oneOf" in node:
                return self._convert_choice(node["oneOf"], "ONE")
            if "anyOf" in node:
                return self._convert_choice(node["anyOf"], "ANY")
            if "allOf" in node:
                return self._convert_allof(node["allOf"])

            typ = node.get("type")
            if isinstance(typ, list):
                nt = self.fresh("UNION")
                for t in typ:
                    sub = self._convert_node({"type": t})
                    if sub:
                        self.cfg.add(nt, [Nonterminal(sub)])
                return nt

            if typ == "object":
                return self._convert_object(node)
            elif typ == "array":
                return self._convert_array(node)
            elif typ == "string":
                return self._convert_primitive("STR", "<string>")
            elif typ == "integer":
                return self._convert_primitive("INT", "<integer>")
            elif typ == "number":
                return self._convert_primitive("NUM", "<number>")
            elif typ == "boolean":
                return self._convert_boolean()
            elif typ == "null":
                nt = self.fresh("NULL")
                self.cfg.add(nt, [Terminal("null")])
                return nt

            if "properties" in node:
                return self._convert_object(node)
            if "items" in node:
                return self._convert_array(node)

            nt = self.fresh("VAL")
            self.cfg.add(nt, [Terminal("<value>")])
            return nt
        finally:
            self._depth -= 1

    def _convert_primitive(self, prefix, token):
        nt = self.fresh(prefix)
        self.cfg.add(nt, [Terminal(token)])
        return nt

    def _convert_boolean(self):
        nt = self.fresh("BOOL")
        self.cfg.add(nt, [Terminal("true")])
        self.cfg.add(nt, [Terminal("false")])
        return nt

    def _convert_object(self, node):
        props = node.get("properties", {})
        additional = node.get("additionalProperties")

        if not props and not additional:
            nt = self.fresh("OBJ")
            self.cfg.add(nt, [Terminal("{"), Terminal("}")])
            return nt

        nt = self.fresh("OBJ")
        prop_names = list(props.keys())

        if not prop_names:
            self.cfg.add(nt, [Terminal("{"), Terminal("}")])
            if additional and isinstance(additional, dict):
                addl_nt = self._convert_node(additional)
                if addl_nt:
                    kvp = self.fresh("ADDLKV")
                    self.cfg.add(kvp, [
                        Terminal("<key>"), Terminal(":"),
                        Nonterminal(addl_nt)
                    ])
                    rep = self.fresh("ADDLREP")
                    self.cfg.add(rep, [Nonterminal(kvp)])
                    self.cfg.add(rep, [
                        Nonterminal(kvp), Terminal(","),
                        Nonterminal(rep)
                    ])
                    self.cfg.add(nt, [
                        Terminal("{"), Nonterminal(rep), Terminal("}")
                    ])
            return nt


        chain_nts = []
        for i, pname in enumerate(prop_names):
            chain_nts.append(self.fresh("CH"))

        for i, pname in enumerate(prop_names):
            val_schema = props[pname]
            val_nt = self._convert_node(val_schema)

            rhs = [Terminal(f'"{pname}"'), Terminal(":")]
            if val_nt:
                rhs.append(Nonterminal(val_nt))

            if i < len(prop_names) - 1:
                rhs.append(Terminal(","))
                rhs.append(Nonterminal(chain_nts[i + 1]))
                self.cfg.add(chain_nts[i], rhs)
            else:
                rhs.append(Terminal("}"))
                self.cfg.add(chain_nts[i], rhs)

        self.cfg.add(nt, [Terminal("{"), Nonterminal(chain_nts[0])])

        if additional and isinstance(additional, dict):
            addl_nt = self._convert_node(additional)
            if addl_nt:
                last_chain = chain_nts[-1]
                ext = self.fresh("EXT")
                self.cfg.add(ext, [
                    Terminal(","), Terminal("<key>"),
                    Terminal(":"), Nonterminal(addl_nt)
                ])
                rep = self.fresh("EXTREP")
                self.cfg.add(rep, [Nonterminal(ext)])
                self.cfg.add(rep, [
                    Nonterminal(ext), Nonterminal(rep)
                ])

        return nt

    def _convert_array(self, node):
        items = node.get("items")
        nt = self.fresh("ARR")

        if not items:
            self.cfg.add(nt, [Terminal("["), Terminal("]")])
            return nt

        item_nt = self._convert_node(items)
        if not item_nt:
            self.cfg.add(nt, [Terminal("["), Terminal("]")])
            return nt

        items_nt = self.fresh("ITEMS")
        self.cfg.add(items_nt, [Nonterminal(item_nt)])
        self.cfg.add(items_nt, [
            Nonterminal(item_nt), Terminal(","), Nonterminal(items_nt)
        ])

        self.cfg.add(nt, [Terminal("["), Terminal("]")])
        self.cfg.add(nt, [
            Terminal("["), Nonterminal(items_nt), Terminal("]")
        ])

        prefix = node.get("prefixItems", node.get("items" if isinstance(items, list) else "_skip_"))
        if isinstance(prefix, list):
            tuple_nt = self.fresh("TUPLE")
            chain = []
            for pi in prefix:
                pi_nt = self._convert_node(pi)
                if pi_nt:
                    chain.append(pi_nt)
            if chain:
                rhs = [Terminal("[")]
                for j, c in enumerate(chain):
                    rhs.append(Nonterminal(c))
                    if j < len(chain) - 1:
                        rhs.append(Terminal(","))
                rhs.append(Terminal("]"))
                self.cfg.add(tuple_nt, rhs)

        return nt

    def _convert_choice(self, options, prefix):
        nt = self.fresh(prefix)
        for opt in options:
            sub = self._convert_node(opt)
            if sub:
                self.cfg.add(nt, [Nonterminal(sub)])
        return nt

    def _convert_allof(self, schemas):
        nt = self.fresh("ALL")
        merged = {}
        for s in schemas:
            if isinstance(s, dict):
                for k, v in s.items():
                    if k == "properties" and k in merged:
                        merged[k].update(v)
                    else:
                        merged[k] = v
        sub = self._convert_node(merged)
        if sub:
            self.cfg.add(nt, [Nonterminal(sub)])
        return nt

    def _resolve_ref(self, ref: str):
        if ref in self._ref_cache:
            return self._ref_cache[ref]

        placeholder = self.fresh("REF")
        self._ref_cache[ref] = placeholder

        if ref.startswith("#/"):
            parts = ref[2:].split("/")
            node = self._root_schema
            for p in parts:
                p = p.replace("~1", "/").replace("~0", "~")
                if isinstance(node, dict):
                    node = node.get(p)
                else:
                    node = None
                    break
            if node and isinstance(node, dict):
                sub = self._convert_node(node)
                if sub:
                    self.cfg.add(placeholder, [Nonterminal(sub)])
                    return placeholder

        self.cfg.add(placeholder, [Terminal(f"<ref:{ref}>")])
        return placeholder



@dataclass
class SchemaAnalysis:
    dataset: str
    schema_id: str
    num_productions: int     # |P|
    num_nonterminals: int    # |N|
    num_terminals: int       # |Σ|
    max_rhs_nt: int          # max nonterminals per production RHS
    nonlinear_productions: int
    is_linear: bool
    has_recursion: bool
    has_array: bool
    has_nested_object: bool
    schema_size: int         # raw JSON byte count
    error: str = ""


def detect_features(schema: dict, depth=0) -> dict:
    """Detect structural features of a JSON schema."""
    feats = {"array": False, "nested_object": False, "recursive": False}
    if not isinstance(schema, dict) or depth > 20:
        return feats

    if schema.get("type") == "array" or "items" in schema:
        feats["array"] = True

    props = schema.get("properties", {})
    for v in props.values():
        if isinstance(v, dict):
            if v.get("type") == "object" or "properties" in v:
                feats["nested_object"] = True
            sub = detect_features(v, depth + 1)
            feats["array"] = feats["array"] or sub["array"]
            feats["nested_object"] = feats["nested_object"] or sub["nested_object"]

    if "$ref" in schema:
        feats["recursive"] = True

    for key in ("oneOf", "anyOf", "allOf", "items", "additionalProperties"):
        child = schema.get(key)
        if isinstance(child, dict):
            sub = detect_features(child, depth + 1)
            for k in feats:
                feats[k] = feats[k] or sub[k]
        elif isinstance(child, list):
            for item in child:
                if isinstance(item, dict):
                    sub = detect_features(item, depth + 1)
                    for k in feats:
                        feats[k] = feats[k] or sub[k]

    return feats


def analyze_schema(schema_json: str, dataset: str, schema_id: str) -> SchemaAnalysis:
    """Full analysis pipeline for one schema."""
    try:
        schema = json.loads(schema_json) if isinstance(schema_json, str) else schema_json
    except (json.JSONDecodeError, TypeError) as e:
        return SchemaAnalysis(
            dataset=dataset, schema_id=schema_id,
            num_productions=0, num_nonterminals=0, num_terminals=0,
            max_rhs_nt=0, nonlinear_productions=0, is_linear=True,
            has_recursion=False, has_array=False, has_nested_object=False,
            schema_size=0, error=str(e)
        )

    feats = detect_features(schema)
    converter = SchemaConverter()
    cfg = converter.convert(schema)

    if cfg is None:
        return SchemaAnalysis(
            dataset=dataset, schema_id=schema_id,
            num_productions=0, num_nonterminals=0, num_terminals=0,
            max_rhs_nt=0, nonlinear_productions=0, is_linear=True,
            has_recursion=feats["recursive"], has_array=feats["array"],
            has_nested_object=feats["nested_object"],
            schema_size=len(json.dumps(schema)), error="conversion_failed"
        )

    return SchemaAnalysis(
        dataset=dataset,
        schema_id=schema_id,
        num_productions=cfg.P,
        num_nonterminals=cfg.N,
        num_terminals=len(cfg.terminals),
        max_rhs_nt=cfg.max_rhs_nt,
        nonlinear_productions=cfg.nonlinear_count,
        is_linear=cfg.is_linear,
        has_recursion=feats["recursive"],
        has_array=feats["array"],
        has_nested_object=feats["nested_object"],
        schema_size=len(json.dumps(schema))
    )



def load_jsonschemabench():
    """Load JSONSchemaBench from HuggingFace."""
    try:
        from datasets import load_dataset
        ds = load_dataset("epfl-dlab/JSONSchemaBench")
        return ds
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        print("Falling back to SchemaStore catalog...")
        return None


def load_schemastore_fallback():
    """Fallback: fetch SchemaStore catalog directly."""
    import urllib.request
    url = "https://www.schemastore.org/api/json/catalog.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            catalog = json.loads(resp.read().decode())
        schemas = []
        for entry in catalog.get("schemas", [])[:200]:
            schema_url = entry.get("url", "")
            if not schema_url:
                continue
            try:
                with urllib.request.urlopen(schema_url, timeout=10) as r:
                    schema_text = r.read().decode()
                schemas.append({
                    "name": entry.get("name", ""),
                    "schema": schema_text,
                    "source": "schemastore"
                })
                print(f"  fetched: {entry.get('name', '')}")
            except Exception:
                continue
        return schemas
    except Exception as e:
        print(f"SchemaStore fallback failed: {e}")
        return []



def print_report(results: list):
    """Print manuscript-ready tables."""
    valid = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    print("\n" + "=" * 72)
    print("  EMPIRICAL ANALYSIS: JSON Schema → CFG Classification")
    print("  For: Validation-Gated Graph Reachability (Alpay)")
    print("=" * 72)

    datasets = sorted(set(r.dataset for r in valid))
    print(f"\nTotal schemas analyzed: {len(results)}")
    print(f"  Successful conversions: {len(valid)}")
    print(f"  Failed/skipped: {len(failed)}")

    print("\n--- Table A: Grammar Class Distribution by Dataset ---")
    print(f"{'Dataset':<20} {'Total':>6} {'Linear':>7} {'General':>8} "
          f"{'%Linear':>8} {'Avg|P|':>7} {'Avg|N|':>7} {'MaxNT':>6}")
    print("-" * 72)

    total_lin = 0
    total_gen = 0
    all_P = []
    all_N = []

    for ds in datasets:
        ds_results = [r for r in valid if r.dataset == ds]
        n_total = len(ds_results)
        n_lin = sum(1 for r in ds_results if r.is_linear)
        n_gen = n_total - n_lin
        total_lin += n_lin
        total_gen += n_gen

        avg_p = sum(r.num_productions for r in ds_results) / max(n_total, 1)
        avg_n = sum(r.num_nonterminals for r in ds_results) / max(n_total, 1)
        max_nt = max((r.max_rhs_nt for r in ds_results), default=0)
        pct = (n_lin / n_total * 100) if n_total > 0 else 0

        all_P.extend(r.num_productions for r in ds_results)
        all_N.extend(r.num_nonterminals for r in ds_results)

        print(f"{ds:<20} {n_total:>6} {n_lin:>7} {n_gen:>8} "
              f"{pct:>7.1f}% {avg_p:>7.1f} {avg_n:>7.1f} {max_nt:>6}")

    total = total_lin + total_gen
    pct_total = (total_lin / total * 100) if total > 0 else 0
    print("-" * 72)
    print(f"{'TOTAL':<20} {total:>6} {total_lin:>7} {total_gen:>8} "
          f"{pct_total:>7.1f}%")

    print("\n--- Table B: Non-linearity Sources ---")
    gen_results = [r for r in valid if not r.is_linear]
    n_with_array = sum(1 for r in gen_results if r.has_array)
    n_with_nested = sum(1 for r in gen_results if r.has_nested_object)
    n_with_rec = sum(1 for r in gen_results if r.has_recursion)
    n_gen = len(gen_results)

    print(f"  Non-linear grammars total: {n_gen}")
    if n_gen > 0:
        print(f"  Contains array type:       {n_with_array:>6} "
              f"({n_with_array/n_gen*100:.1f}%)")
        print(f"  Contains nested object:    {n_with_nested:>6} "
              f"({n_with_nested/n_gen*100:.1f}%)")
        print(f"  Contains $ref (recursion): {n_with_rec:>6} "
              f"({n_with_rec/n_gen*100:.1f}%)")

    print("\n--- Table C: Grammar Size Distribution ---")
    if all_P:
        all_P_sorted = sorted(all_P)
        all_N_sorted = sorted(all_N)
        n = len(all_P_sorted)
        print(f"  |P| — min: {all_P_sorted[0]}, "
              f"median: {all_P_sorted[n//2]}, "
              f"mean: {sum(all_P)/n:.1f}, "
              f"p95: {all_P_sorted[int(n*0.95)]}, "
              f"max: {all_P_sorted[-1]}")
        print(f"  |N| — min: {all_N_sorted[0]}, "
              f"median: {all_N_sorted[n//2]}, "
              f"mean: {sum(all_N)/n:.1f}, "
              f"p95: {all_N_sorted[int(n*0.95)]}, "
              f"max: {all_N_sorted[-1]}")

    print("\n--- Table D: Manuscript Summary (copy-paste to LaTeX) ---")
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"Grammar Class & Count & Percentage & Avg $|P|$ \\")
    print(r"\midrule")
    lin_results = [r for r in valid if r.is_linear]
    avg_p_lin = sum(r.num_productions for r in lin_results) / max(len(lin_results), 1)
    avg_p_gen = sum(r.num_productions for r in gen_results) / max(len(gen_results), 1)
    print(f"Linear & {total_lin} & {pct_total:.1f}\\% & {avg_p_lin:.1f} \\\\")
    print(f"General CFG & {total_gen} & {100-pct_total:.1f}\\% & {avg_p_gen:.1f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    print("\n--- Table E: Schema Size vs Grammar Class ---")
    lin_sizes = [r.schema_size for r in valid if r.is_linear]
    gen_sizes = [r.schema_size for r in valid if not r.is_linear]
    if lin_sizes:
        print(f"  Linear schemas — avg size: {sum(lin_sizes)/len(lin_sizes):.0f} bytes")
    if gen_sizes:
        print(f"  General schemas — avg size: {sum(gen_sizes)/len(gen_sizes):.0f} bytes")


def save_csv(results: list, path="schema_analysis.csv"):
    """Save full results to CSV for further analysis."""
    fields = [
        "dataset", "schema_id", "num_productions", "num_nonterminals",
        "num_terminals", "max_rhs_nt", "nonlinear_productions",
        "is_linear", "has_recursion", "has_array", "has_nested_object",
        "schema_size", "error"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"\nFull results saved to: {path}")



def main():
    print("Loading JSONSchemaBench...")
    ds = load_jsonschemabench()

    results = []

    if ds is not None:
        for split_name in ds:
            split = ds[split_name]
            print(f"\nProcessing split: {split_name} ({len(split)} schemas)")
            for i, row in enumerate(split):
                schema_text = row.get("schema", row.get("json_schema", ""))
                sid = row.get("id", row.get("name", f"{split_name}_{i}"))
                result = analyze_schema(schema_text, split_name, str(sid))
                results.append(result)
                if (i + 1) % 500 == 0:
                    print(f"  ...{i+1}/{len(split)}")
    else:
        print("\nUsing SchemaStore fallback...")
        schemas = load_schemastore_fallback()
        for i, entry in enumerate(schemas):
            result = analyze_schema(
                entry["schema"], "schemastore", entry["name"]
            )
            results.append(result)
            if (i + 1) % 50 == 0:
                print(f"  ...{i+1}/{len(schemas)}")

    if not results:
        print("No schemas processed. Exiting.")
        return

    print_report(results)
    save_csv(results)

    print("\n--- Example: First linear schema ---")
    for r in results:
        if r.is_linear and r.num_productions > 2 and not r.error:
            print(f"  {r.dataset}/{r.schema_id}: "
                  f"|P|={r.num_productions}, |N|={r.num_nonterminals}")
            break

    print("\n--- Example: First non-linear schema ---")
    for r in results:
        if not r.is_linear and not r.error:
            print(f"  {r.dataset}/{r.schema_id}: "
                  f"|P|={r.num_productions}, |N|={r.num_nonterminals}, "
                  f"max_rhs_nt={r.max_rhs_nt}, "
                  f"array={r.has_array}, nested={r.has_nested_object}")
            break


if __name__ == "__main__":
    main()
