from agent_graph.kg.confidence_toolkit import combine_confidence, EmpiricalCIResult


def test_textual_only_returns_textual():
    assert combine_confidence(0.7) == 0.7


def test_clamps_out_of_range_textual():
    assert combine_confidence(1.5) == 1.0
    assert combine_confidence(-0.2) == 0.0


def test_underpowered_empirical_is_dropped():
    r = EmpiricalCIResult("underpowered", "zero-flow-ci", 50)
    # non-informative -> reduces to textual-only
    assert combine_confidence(0.7, empirical=r) == 0.7


def test_supports_boosts_and_refutes_lowers():
    base = 0.5
    sup = EmpiricalCIResult("supports", "zero-flow-ci", 500)
    ref = EmpiricalCIResult("refutes", "zero-flow-ci", 500)
    assert combine_confidence(base, empirical=sup) > base
    assert combine_confidence(base, empirical=ref) < base


def test_all_three_legs_weighted_and_in_unit_interval():
    r = EmpiricalCIResult("supports", "zero-flow-ci", 500)
    c = combine_confidence(0.6, structural=0.8, empirical=r)
    assert 0.0 <= c <= 1.0
    # 0.5*0.6 + 0.25*0.8 + 0.25*0.9 over total weight 1.0
    assert abs(c - (0.5 * 0.6 + 0.25 * 0.8 + 0.25 * 0.9)) < 1e-9


def test_structural_only_leg_renormalizes():
    # textual + structural, empirical absent: weights 0.5 + 0.25 renormalized
    c = combine_confidence(0.4, structural=1.0)
    assert abs(c - (0.5 * 0.4 + 0.25 * 1.0) / 0.75) < 1e-9
