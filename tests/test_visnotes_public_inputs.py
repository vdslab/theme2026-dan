import unittest

from experiments.visnotes_public_validation_01.prepare_inputs import dblp_input, maximal_groups


def er_with_edge(edge):
    return (
        "Nary Relationship Data 2D\n"
        "Comma Delimiter\n"
        "Relationship Colors Saved\n"
        "3,1\n"
        "0,Alpha,0,0\n"
        "1,Beta,1,0\n"
        "2,Gamma,0,1\n"
        f"{edge}\n"
    ).encode()


class VisNotesPublicInputsTests(unittest.TestCase):
    def test_longest_author_suffix_preserves_commas_and_boundary_order(self):
        data = dblp_input(er_with_edge("0,10,20,30,Comma, in title,3,1,2,0"))
        self.assertEqual(data["hyperedges"][0]["label"], "Comma, in title")
        self.assertEqual(data["hyperedges"][0]["nodes"], [1, 2, 0])

    def test_invalid_longer_suffix_does_not_turn_into_a_different_paper(self):
        with self.assertRaisesRegex(ValueError, "Unknown or repeated"):
            dblp_input(er_with_edge("0,10,20,30,Paper,3,99,1,2"))

    def test_unary_orientation_not_parsed_as_a_vertex(self):
        data = dblp_input(er_with_edge("0,10,20,30,Editorial,1,2,-0.25"))
        self.assertEqual(data["hyperedges"][0]["nodes"], [2])
        self.assertEqual(data["hyperedges"][0]["source_monogon_angle"], -0.25)

    def test_only_maximal_higher_order_groups_with_deterministic_order(self):
        contacts = [
            (3, 4), (1, 3), (1, 4), (1, 2), (2, 3),
            (2, 4), (4, 5), (5, 6), (4, 6), (6, 7),
        ]
        self.assertEqual(maximal_groups(reversed(contacts)), [(1, 2, 3, 4), (4, 5, 6)])
