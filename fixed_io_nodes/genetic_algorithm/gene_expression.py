"""
Gene-expression: suppress certain hyperparameter combinations so they are never trained.
Parses forbid_combinations and rules from config; returns a predicate is_suppressed(individual).
"""

from typing import Any, Callable, Dict, List


def _matcher_from_condition(condition: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    """Match when individual satisfies all keys: value can be single (==) or list (in)."""
    def matches(individual: Dict[str, Any]) -> bool:
        for k, v in condition.items():
            val = individual.get(k)
            if isinstance(v, list):
                if val not in v:
                    return False
            else:
                if val != v:
                    return False
        return True
    return matches


def build_suppress_predicate(gene_expression_raw: Dict[str, Any]) -> Callable[[Dict[str, Any]], bool]:
    """
    Build is_suppressed(individual) from config gene_expression.
    Supports forbid_combinations (list of dicts) and rules (list of { when: {...} }).
    Missing/empty gene_expression -> never suppress.
    """
    if not gene_expression_raw or not isinstance(gene_expression_raw, dict):
        return lambda _: False

    matchers: List[Callable[[Dict[str, Any]], bool]] = []

    for item in gene_expression_raw.get("forbid_combinations") or []:
        if isinstance(item, dict) and item:
            matchers.append(_matcher_from_condition(item))

    for rule in gene_expression_raw.get("rules") or []:
        if isinstance(rule, dict):
            when = rule.get("when")
            if isinstance(when, dict) and when:
                matchers.append(_matcher_from_condition(when))

    if not matchers:
        return lambda _: False

    def is_suppressed(individual: Dict[str, Any]) -> bool:
        return any(m(individual) for m in matchers)

    return is_suppressed
